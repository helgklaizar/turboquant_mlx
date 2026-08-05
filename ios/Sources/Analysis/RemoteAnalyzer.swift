import Foundation

/// Talks to a Mac on the same network running the TurboQuant backend.
///
/// Two transports, tried in order:
///
/// 1. `POST /v1/analyze` — the structured endpoint from
///    `scripts/run_assistant_server.py`. The server owns the prompt and returns
///    the JSON schema directly.
/// 2. `POST /v1/chat/completions` — the plain OpenAI-compatible surface exposed by
///    `scripts/run_server.py`. The prompt is built here and the JSON is fished out
///    of the assistant message.
///
/// The fallback means the app is useful against any OpenAI-compatible endpoint,
/// not only ours.
final class RemoteAnalyzer: Analyzer {

    private let baseURL: URL
    private let model: String
    private let session: URLSession
    /// Set once `/v1/analyze` is known to be missing, so we stop probing it.
    private var structuredEndpointAvailable = true

    init(baseURLString: String, model: String, timeout: TimeInterval = 120) throws {
        let trimmed = baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme != nil, url.host != nil else {
            throw AnalyzerError.badURL(baseURLString)
        }
        self.baseURL = url
        self.model = model

        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = timeout
        configuration.waitsForConnectivity = false
        self.session = URLSession(configuration: configuration)
    }

    func analyze(_ request: AnalysisRequest) async throws -> AnalysisResult {
        if structuredEndpointAvailable {
            do {
                return try await analyzeStructured(request)
            } catch AnalyzerError.http(let code, _) where code == 404 || code == 405 {
                structuredEndpointAvailable = false
            }
        }
        return try await analyzeViaChatCompletions(request)
    }

    /// `GET /health` — used by Settings to tell "wrong address" from "model still loading".
    func checkHealth() async throws -> String {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("health"))
        urlRequest.timeoutInterval = 10
        let (data, response) = try await session.data(for: urlRequest)
        try Self.validate(response, data: data)
        return String(data: data, encoding: .utf8) ?? ""
    }

    // MARK: - Transports

    private func analyzeStructured(_ request: AnalysisRequest) async throws -> AnalysisResult {
        let body: [String: Any] = [
            "transcript": request.transcript,
            "previous_summary": request.previousSummary ?? "",
            "known_insights": request.knownInsights,
            "language": request.language,
            "mode": request.mode.rawValue
        ]
        let data = try await post(path: "v1/analyze", body: body)
        return try JSONDecoder().decode(AnalysisPayload.self, from: data).toResult()
    }

    private func analyzeViaChatCompletions(_ request: AnalysisRequest) async throws -> AnalysisResult {
        let body: [String: Any] = [
            "model": model,
            "temperature": 0.2,
            "max_tokens": 900,
            "messages": [
                ["role": "system", "content": AnalysisPrompt.system],
                ["role": "user", "content": AnalysisPrompt.user(for: request)]
            ]
        ]
        let data = try await post(path: "v1/chat/completions", body: body)

        struct ChatResponse: Decodable {
            struct Choice: Decodable {
                struct Message: Decodable { let content: String? }
                let message: Message?
            }
            let choices: [Choice]
        }

        let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
        let content = decoded.choices.first?.message?.content ?? ""
        return try AnalysisPayload.decode(fromModelOutput: content).toResult()
    }

    // MARK: - Plumbing

    private func post(path: String, body: [String: Any]) async throws -> Data {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent(path))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: urlRequest)
        try Self.validate(response, data: data)
        return data
    }

    private static func validate(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw AnalyzerError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
    }
}
