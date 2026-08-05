import AVFoundation
import Combine
import Foundation
import SwiftUI

enum RecorderState: Equatable {
    case idle
    case preparing
    case listening
    case finishing
    case failed(String)

    var isActive: Bool {
        self == .listening || self == .preparing
    }
}

/// Wires capture → transcription → analysis → storage, and exposes the whole thing
/// to SwiftUI as one object.
@MainActor
final class RecorderViewModel: ObservableObject {

    @Published private(set) var state: RecorderState = .idle
    /// Speech already committed by the recognizer.
    @Published private(set) var transcript: String = ""
    /// The hypothesis currently in flight, shown greyed out under the transcript.
    @Published private(set) var partial: String = ""
    @Published private(set) var level: Float = 0
    @Published private(set) var elapsed: TimeInterval = 0

    let pipeline = InsightPipeline()

    private let capture = AudioCaptureService()
    private var transcriber: SpeechTranscriptionService?
    private var session: Session?
    private var timer: Timer?
    private var settings: AppSettings?
    private var store: SessionStore?
    private var cancellables = Set<AnyCancellable>()

    init() {
        // A nested ObservableObject does not propagate on its own — SwiftUI would
        // never redraw when the pipeline publishes a new insight. Forward it.
        pipeline.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    var liveText: String {
        partial.isEmpty ? transcript : (transcript.isEmpty ? partial : transcript + " " + partial)
    }

    func bind(settings: AppSettings, store: SessionStore) {
        self.settings = settings
        self.store = store
    }

    // MARK: - Control

    func toggle() async {
        if state.isActive {
            await stop()
        } else {
            await start()
        }
    }

    func start() async {
        guard let settings, !state.isActive else { return }
        state = .preparing

        guard await AudioCaptureService.requestPermission() else {
            state = .failed(AudioCaptureError.microphoneDenied.localizedDescription)
            return
        }
        guard await SpeechTranscriptionService.requestPermission() else {
            state = .failed(TranscriptionError.notAuthorized.localizedDescription)
            return
        }

        transcript = ""
        partial = ""
        elapsed = 0
        pipeline.reset()
        pipeline.configure(
            analyzer: settings.makeAnalyzer(),
            mode: settings.mode,
            language: settings.localeIdentifier
        )

        var newSession = Session(mode: settings.mode)
        let audioURL: URL?
        if settings.keepAudio, let store {
            let name = "\(newSession.id.uuidString).caf"
            newSession.audioFileName = name
            audioURL = store.audioURL(for: name)
        } else {
            audioURL = nil
        }
        session = newSession

        let speech = SpeechTranscriptionService(localeIdentifier: settings.localeIdentifier)
        speech.onPartial = { [weak self] text in
            Task { @MainActor in self?.partial = text }
        }
        speech.onFinalSegment = { [weak self] text in
            Task { @MainActor in self?.commit(segment: text) }
        }
        speech.onError = { [weak self] error in
            Task { @MainActor in self?.pipeline.lastError = error.localizedDescription }
        }
        transcriber = speech

        capture.onBuffer = { [weak speech] buffer in
            speech?.append(buffer)
        }
        capture.onLevel = { [weak self] value in
            Task { @MainActor in self?.level = value }
        }

        do {
            try speech.start()
            try capture.start(recordingTo: audioURL)
        } catch {
            capture.stop()
            speech.stop()
            transcriber = nil
            state = .failed(error.localizedDescription)
            return
        }

        state = .listening
        startTimer()
    }

    func stop() async {
        guard state.isActive else { return }
        state = .finishing
        stopTimer()

        capture.stop()
        transcriber?.stop()
        transcriber = nil
        level = 0

        // A final hypothesis may land after stop(); fold it in before analysing.
        if !partial.isEmpty {
            commit(segment: partial)
            partial = ""
        }

        await pipeline.finish(fullTranscript: transcript)

        if var finished = session {
            finished.endedAt = Date()
            finished.transcript = transcript
            finished.summary = pipeline.summary
            finished.insights = pipeline.insights
            if !finished.transcript.isEmpty || !finished.insights.isEmpty {
                store?.save(finished)
            }
            session = finished
        }

        state = .idle
    }

    func clearError() {
        if case .failed = state { state = .idle }
        pipeline.lastError = nil
    }

    // MARK: - Internals

    private func commit(segment: String) {
        let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        transcript += (transcript.isEmpty ? "" : " ") + trimmed
        partial = ""
        pipeline.append(segment: trimmed)
    }

    private func startTimer() {
        let started = Date()
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.elapsed = Date().timeIntervalSince(started) }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}
