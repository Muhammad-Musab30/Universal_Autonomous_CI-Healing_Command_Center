Here is the Master Architecture Flowchart for the upgraded "RepairIQ" Universal CI-Healing Agent:


flowchart TD
    subgraph Input["1. Universal Trigger Layer (Any Code or Repo)"]
        A1["Mode A: Paste ANY GitHub Repo URL + Branch"]
        A2["Mode B: Live Code & Test Sandbox Editor (Paste Any Code)"]
        A3["Mode C: GitHub Actions Webhook / Failed Run ID"]
        A4["Mode D: 3 Seeded Demo Repos (Env Bug, Flaky Test, Logic Bug)"]
    end

    subgraph Ingest["2. Dynamic Ingestion & Evidence Graph Builder"]
        B1["Clone Repo / Write Custom Files into Isolated Sandbox"]
        B2["Auto-Detect Runner (pytest / unittest / npm test / ruff)"]
        B3["Execute Initial Run & Capture Raw Logs + Exit Code"]
        B4["Extract Stack Trace, Implicated Files, & Git Diff"]
        B5["Build Clickable Evidence Graph + Check Team Repair Memory"]
    end

    subgraph Router["3. Nebius Token Factory Two-Model Router"]
        C1["Nemotron Nano / Super: Classify Failure Taxonomy"]
        C2{"Failure Category?"}
        C3["Flaky Test Intelligence: Rerun 3x to detect nondeterminism"]
        C4["Config / Env / Workflow Mode: Propose No-Code Fix"]
        C5["Nemotron Ultra: Deep Diagnosis, Hypotheses & Tool Plan"]
    end

    subgraph Loop["4. Iterative Agent State Machine (Max 3-5 Attempts)"]
        D1["Plan Max 3 Safe Actions (Inspect File, Apply Minimal Patch, Run Test)"]
        D2["Execute Allowlisted Command in Sandbox"]
        D3["Verification: Reproduce Failure -> Apply Patch -> Rerun Failing Test -> Full Suite"]
        D4{"Tests Pass & Minimal Diff?"}
        D5["Feed New Error Logs Back to Nemotron Ultra"]
        D6{"Iteration <= 5?"}
    end

    subgraph Risk["5. Confidence Scorer & Risk-Tiered Autonomy"]
        E1["Calculate Repair Confidence Score (0-100)"]
        E2{"Risk Tier?"}
        E3["90-100 High Confidence: Ready for 1-Click PR"]
        E4["50-89 Medium Confidence: Diagnosis + Suggested Patch"]
        E5["0-49 Low Confidence / Unsafe Diff: Escalate to Human Only"]
    end

    subgraph Delivery["6. Delivery & Team Memory Loop"]
        F1["Interactive Review Dashboard (Clickable Graph, Diff, Telemetry)"]
        F2["Human Approval -> Create Real GitHub Branch, Commit & Open PR"]
        F3["Save Outcome & Reviewer Conventions to Team Repair Memory"]
    end

    A1 & A2 & A3 & A4 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> C1 --> C2
    C2 -->|Flaky Test| C3 --> E1
    C2 -->|Missing Env / Config| C4 --> E1
    C2 -->|Code / Logic / Import / Type Bug| C5 --> D1
    D1 --> D2 --> D3 --> D4
    D4 -->|No| D5 --> D6
    D6 -->|Yes| C5
    D6 -->|No: Stop & Escalate| E5
    D4 -->|Yes| E1 --> E2
    E2 -->|High| E3 --> F1
    E2 -->|Medium| E4 --> F1
    E2 -->|Low| E5 --> F1
    F1 --> F2 --> F3