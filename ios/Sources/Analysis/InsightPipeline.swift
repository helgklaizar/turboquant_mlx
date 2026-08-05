import Foundation

/// Decides *when* to analyse and merges what comes back.
///
/// In `.button` mode this runs exactly once, on stop, over the whole transcript.
/// In `.realtime` mode it batches finalized speech segments and fires whenever
/// enough new text has accumulated — one request in flight at a time, so a slow
/// model backs the pipeline up instead of stacking requests on it.
@MainActor
final class InsightPipeline: ObservableObject {

    @Published private(set) var insights: [Insight] = []
    @Published private(set) var summary: String = ""
    @Published private(set) var isAnalyzing = false
    @Published var lastError: String?

    /// Enough new speech to be worth a round trip.
    private let minCharsForRealtimePass = 220
    /// ...unless this much time has passed, then a shorter fragment will do.
    private let maxSilentInterval: TimeInterval = 45
    private let minCharsAfterInterval = 60
    /// Never hammer the backend faster than this.
    private let minInterval: TimeInterval = 12

    private var analyzer: Analyzer?
    private var pending = ""
    private var seenKeys: Set<String> = []
    private var lastRun = Date.distantPast
    private var inFlight: Task<Void, Never>?

    var mode: CaptureMode = .button
    var language: String = "ru-RU"

    func configure(analyzer: Analyzer?, mode: CaptureMode, language: String) {
        self.analyzer = analyzer
        self.mode = mode
        self.language = language
    }

    func reset() {
        inFlight?.cancel()
        inFlight = nil
        insights = []
        summary = ""
        pending = ""
        seenKeys = []
        lastRun = .distantPast
        isAnalyzing = false
        lastError = nil
    }

    /// Feed a committed transcript segment.
    func append(segment: String) {
        let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        pending += (pending.isEmpty ? "" : " ") + trimmed

        guard mode == .realtime else { return }
        if shouldRunRealtimePass() {
            run(on: pending, clearingPending: true)
        }
    }

    /// Called when recording stops. In button mode this is the only pass, so it
    /// looks at the full transcript rather than the tail.
    func finish(fullTranscript: String) async {
        switch mode {
        case .button:
            pending = ""
            await runAndWait(on: fullTranscript)
        case .realtime:
            let tail = pending
            pending = ""
            if tail.count >= 20 {
                await runAndWait(on: tail)
            } else {
                await inFlight?.value
            }
        }
    }

    // MARK: - Triggering

    private func shouldRunRealtimePass() -> Bool {
        guard inFlight == nil else { return false }
        let elapsed = Date().timeIntervalSince(lastRun)
        if elapsed < minInterval { return false }
        if pending.count >= minCharsForRealtimePass { return true }
        return elapsed >= maxSilentInterval && pending.count >= minCharsAfterInterval
    }

    private func run(on text: String, clearingPending: Bool) {
        guard inFlight == nil else { return }
        if clearingPending { pending = "" }
        inFlight = Task { [weak self] in
            await self?.performAnalysis(on: text)
            self?.inFlight = nil
        }
    }

    private func runAndWait(on text: String) async {
        await inFlight?.value
        inFlight = nil
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        await performAnalysis(on: text)
    }

    // MARK: - Execution

    private func performAnalysis(on text: String) async {
        guard let analyzer else {
            lastError = AnalyzerError.notConfigured.localizedDescription
            return
        }

        lastRun = Date()
        isAnalyzing = true
        defer { isAnalyzing = false }

        let request = AnalysisRequest(
            transcript: text,
            previousSummary: summary.isEmpty ? nil : summary,
            knownInsights: insights.map(\.text),
            language: language,
            mode: mode
        )

        do {
            let result = try await analyzer.analyze(request)
            guard !Task.isCancelled else { return }
            merge(result)
            lastError = nil
        } catch is CancellationError {
            return
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func merge(_ result: AnalysisResult) {
        if !result.summary.isEmpty {
            summary = result.summary
        }
        for insight in result.insights where seenKeys.insert(insight.dedupeKey).inserted {
            insights.append(insight)
        }
    }
}
