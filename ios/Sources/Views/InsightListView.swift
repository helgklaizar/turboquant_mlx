import SwiftUI

struct InsightListView: View {
    let summary: String
    let insights: [Insight]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if !summary.isEmpty {
                    Text(summary)
                        .font(.callout)
                        .textSelection(.enabled)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
                }

                ForEach(insights) { insight in
                    InsightRow(insight: insight)
                }
            }
            .padding()
        }
    }
}

struct InsightRow: View {
    let insight: Insight

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: insight.kind.symbol)
                .font(.title3)
                .frame(width: 26)
                .foregroundStyle(.tint)

            VStack(alignment: .leading, spacing: 4) {
                Text(insight.text)
                    .font(.body)
                    .textSelection(.enabled)

                HStack(spacing: 8) {
                    Text(insight.kind.title)
                    if let who = insight.who { Text("· \(who)") }
                    if let due = insight.due { Text("· \(due)") }
                    if insight.confidence < 0.5 {
                        Text("· uncertain")
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
