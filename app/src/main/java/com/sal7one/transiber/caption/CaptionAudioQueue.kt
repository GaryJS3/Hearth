package com.sal7one.transiber.caption

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.Channel

/**
 * Non-blocking capture ingress for legacy/local/cloud engines.
 * The batch Whisper path can drop stale waiting frames rather than ending capture;
 * all other engines retain the original fail-on-overload behavior.
 */
internal enum class AudioOverloadPolicy { FAIL, DROP_OLDEST }

internal class CaptionAudioQueue(
    private val maxQueuedSamples: Int = 16_000,
    private val overloadPolicy: AudioOverloadPolicy = AudioOverloadPolicy.FAIL,
) {
    init { require(maxQueuedSamples in 1..48_000) }
    private val lock = Any()
    private var pending = 0
    private var dropped = 0L
    private var closed = false
    private var consuming = false
    private var error: Throwable? = null
    private val frames = Channel<ShortArray>(Channel.UNLIMITED, onUndeliveredElement = ::completed)

    val failure: Throwable? get() = synchronized(lock) { error }
    val pendingSamples: Int get() = synchronized(lock) { pending }
    val droppedSamples: Long get() = synchronized(lock) { dropped }

    /** Copy only accepted samples: the recorder may reuse its array as soon as this returns. */
    fun tryAccept(samples: ShortArray, sampleCount: Int): Boolean = synchronized(lock) {
        if (closed) return false
        if (sampleCount !in 1..samples.size || sampleCount > 16_000) {
            fail(IllegalArgumentException("Caption audio frame must contain 1..16000 samples"))
            return false
        }
        if (sampleCount > maxQueuedSamples - pending) {
            if (overloadPolicy == AudioOverloadPolicy.FAIL) {
                fail(IllegalStateException("Audio processing cannot keep up: $pending samples pending; queue limit $maxQueuedSamples"))
                return false
            }
            // Never interrupt an in-flight engine call. Discard only queued frames.
            while (sampleCount > maxQueuedSamples - pending) {
                val stale = frames.tryReceive().getOrNull() ?: break
                pending -= stale.size
                dropped += stale.size
            }
            // If the in-flight frame still fills the budget, discard incoming PCM
            // without shutting down this capture generation.
            if (sampleCount > maxQueuedSamples - pending) {
                dropped += sampleCount
                return false
            }
        }
        val copy = samples.copyOf(sampleCount)
        pending += sampleCount
        check(frames.trySend(copy).isSuccess)
        true
    }

    /** One ordered inference lane; errors remain errors and retire all queued input. */
    suspend fun consume(process: suspend (ShortArray) -> Unit) {
        synchronized(lock) { check(!consuming) { "Caption audio already has a consumer" }; consuming = true }
        try {
            for (samples in frames) {
                try { process(samples) } finally { completed(samples) }
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (cause: Throwable) {
            synchronized(lock) { if (error == null) error = cause }
            throw cause
        } finally { cancel() }
    }

    /** Source ended normally: deliver every accepted frame before finalizing the recognizer. */
    fun finish() = synchronized(lock) { closed = true; frames.close(); Unit }

    /** Clear/Stop/new generation: discard waiting input; the caller joins in-flight inference. */
    fun cancel() = synchronized(lock) { closed = true; frames.cancel() }

    private fun completed(samples: ShortArray) = synchronized(lock) { pending -= samples.size }
    private fun fail(cause: Throwable) {
        error = cause
        closed = true
        frames.cancel()
    }
}
