# VibeFlow: LLM Fine-Tuning IDE
## Product Requirements Document (PRD)
**Version**: 1.0  
**Last Updated**: November 1, 2025  
**Status**: MVP Ready for Development  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Product Vision & Philosophy](#product-vision--philosophy)
3. [Core Features & User Flows](#core-features--user-flows)
4. [Technical Architecture](#technical-architecture)
5. [UI/UX Specifications](#uiux-specifications)
6. [Data Management & Workflow](#data-management--workflow)
7. [Fine-Tuning Engine](#fine-tuning-engine)
8. [Evaluation & Deployment](#evaluation--deployment)
9. [Analytics & Monitoring](#analytics--monitoring)
10. [Security & Privacy](#security--privacy)
11. [Development Roadmap](#development-roadmap)
12. [Success Metrics](#success-metrics)
13. [Open Source Stack](#open-source-stack)

---

## Executive Summary

### Product Name
**VibeFlow** - An AI-powered IDE for accessible LLM fine-tuning that democratizes model customization for developers, startups, and technical teams without ML expertise.

### Core Value Proposition
"Fine-tune LLMs as easily as writing code" - Transform complex ML workflows into an intuitive, IDE-based experience inspired by Cursor's approach to software development.

### Problem Statement
Based on 30 user interviews (73% adoption validation):
- **Data preparation = #1 barrier** (70% of users): 2-4 weeks formatting data
- **Hyperparameter confusion** (60%): No guidance on configurations
- **Cost surprises** (53%): Unexpected $1-3K GPU bills
- **Deployment gap**: Fine-tuned models sit unused; 84% struggle to deploy
- **Fragmented tools**: Users stitch together 5-7 different platforms

### Solution
A **single, cohesive IDE** built on VS Code (open-source, familiar to developers) that:
1. **Eliminates data prep complexity**: Auto-detection, format conversion, quality scoring
2. **Handles hyperparameter decisions**: AI-powered "Brainstorm Agent" recommends configs
3. **Transparent costs**: Real-time estimation before training starts
4. **Guided workflows**: Step-by-step process with integrated evaluation
5. **One-click deployment**: Models go from training to API in minutes

### Target Users (MVP Priority)
1. **Software Developers** (63% adoption): Add AI to apps without ML degree
2. **Startup Founders** (83% adoption): Custom models for competitive advantage
3. **ML Engineers** (71% adoption): Faster iteration cycles
4. **Product Managers** (88% adoption): Enterprise segment for later

### Success Criteria (First 16 Weeks)
- 10-15 paying beta users within 8 weeks
- 70%+ complete end-to-end fine-tuning workflow
- 60%+ deploy to production
- Reduce time from "idea to deployed model" from 4 weeks to 4 hours
- NPS ≥ 40 (industry standard for developer tools)

---

## Product Vision & Philosophy

### Design Principles

#### 1. **Complexity Abstraction, Not Hiding**
Like Cursor, we don't hide complexity—we intelligently assist at every step.
- Users should understand what's happening
- Advanced controls available but optional
- Educational tooltips explain every decision
- Expert mode for ML engineers

#### 2. **Opinionated Defaults**
We make smart defaults based on community best practices:
- Standard LoRA configs for different model sizes
- Recommended learning rates per task
- Pre-validated dataset quality thresholds
- Pre-configured evaluation benchmarks

#### 3. **Developer Experience First**
- Copy VS Code's keyboard shortcuts and muscle memory
- Git integration for version control of models
- Familiar workflows (train/test/deploy)
- Fast feedback loops (real-time training progress)

#### 4. **Cost Transparency**
- No hidden charges or bill shocks
- Upfront cost estimates before training
- Pay only for actual GPU seconds used
- Ability to pause/resume to manage costs

#### 5. **Iterative Refinement**
- Encourage A/B testing of fine-tuned models
- One-click rollback to previous versions
- Production feedback feeds into retraining loops

### User Personas

#### Persona 1: DevRelai (Developer, 28, Mid-Level)
- **Background**: Full-stack developer at 15-person startup
- **Goal**: Add customer service chatbot to SaaS product
- **Pain Point**: "I found 10 tutorials on fine-tuning but all assume I know PyTorch"
- **Validation Signal**: 63% adoption, willing to pay $399-500/month
- **Key Need**: Guided, handholding workflow with no ML knowledge required

#### Persona 2: FounderSarah (Founder, 34, Technical Co-founder)
- **Background**: Building AI-first B2B SaaS, building custom model for domain adaptation
- **Goal**: Proprietary model that competitors can't easily replicate
- **Pain Point**: "My ML person quit and now I'm stuck—can't afford to hire another"
- **Validation Signal**: 83% adoption, highest willingness to pay
- **Key Need**: Fast iteration cycles, ability to experiment without specialists

#### Persona 3: EngineerAlex (ML Engineer, 31, Senior)
- **Background**: ML at mid-market company, fine-tunes regularly
- **Goal**: 10x faster iteration cycles with cleaner workflows
- **Pain Point**: "I spend 60% of time on data pipeline, 20% on training, 20% on deployment"
- **Validation Signal**: 71% adoption, wants advanced features
- **Key Need**: Automate data pipeline and deployment; focus on model innovation

---

## Core Features & User Flows

### Feature 1: AI-Powered Brainstorm Agent

#### User Story
"As a developer with no ML background, I want the system to recommend a model and configuration based on my use case so I don't have to research which model to choose."

#### Functional Requirements

**1.1 Brainstorm Chat Interface**
- **Component**: Conversational AI assistant (Gemini 2.5 API)
- **Location**: Left sidebar or welcome modal (first launch)
- **Interaction Flow**:
  ```
  User: "I want to fine-tune a model for customer support replies"
  
  System: "Great! Let me ask a few questions:
  1. How many training examples do you have? (100, 1K, 10K, 100K+)
  2. What's your priority: speed, accuracy, or cost?
  3. Do you have GPU access? If not, I'll optimize for CPU."
  
  User: "1K examples, accuracy is key, I have an A100 GPU"
  
  System: "Perfect! I recommend:
  📊 Base Model: Mistral-7B (balanced for customer support)
  ⚙️ Method: QLoRA (efficient on 1 GPU)
  💰 Est. Cost: $12-18 for training
  ⏱️ Est. Time: 2-4 hours
  
  Reasoning: Mistral excels at instruction-following (customer 
  replies need this). QLoRA keeps GPU memory low. Your dataset 
  size supports good generalization without overfitting.
  
  Ready to proceed? [Start] [Adjust] [Learn More]"
  ```

**1.2 Model Selection**
- **Display**: Curated list of open-source models with metadata:
  ```
  Model Name | Params | Type | Speed | Cost | Best For
  Mistral-7B | 7B | Chat | ⚡⚡⚡ | $ | Customer service
  LLaMA-2 | 7B/13B/70B | General | ⚡⚡ | $-$$ | Flexible
  Phi-3 | 3.8B | Small | ⚡⚡⚡⚡ | $ | Edge/Local
  Llama-3.1 | 8B/70B | Chat | ⚡⚡ | $$ | High quality
  ```
- **Actions**:
  - Click model → view details (architecture, performance on benchmarks, sample outputs)
  - Compare 2-3 models side-by-side
  - Filter by: parameter count, speed, cost, task type
  - "Let AI Pick" → Brainstorm Agent recommends

**1.3 Brainstorm Agent Backend Logic**
- **System Prompt**:
  ```
  You are VibeFlow's AI tutor. Your role is to:
  1. Ask clarifying questions about the user's LLM fine-tuning goal
  2. Recommend a base model based on their needs (task, data size, hardware)
  3. Suggest a training method (full, LoRA, QLoRA) 
  4. Estimate cost, time, and success likelihood
  5. Explain your reasoning in plain English
  
  Use this decision tree:
  - Params available < 4GB → Phi-3 + QLoRA
  - Customer service task → Mistral-7B + LoRA
  - Large dataset (100K+) → Llama-70B + LoRA
  - One GPU, accuracy priority → Llama-13B + QLoRA
  
  Always be honest about trade-offs. If a task is hard, say so.
  ```
- **API Integration**: Gemini 2.5 API (free tier allows ~60 req/month; paid scale for production)
- **Fallback**: If Gemini unavailable, show static decision tree UI

#### UI Components
- **Sidebar Component** (150px width, collapsible):
  - Title: "Brainstorm Agent"
  - Chat input: "Ask me about your fine-tuning idea..."
  - Chat history (scrollable, max 10 messages in MVP)
  - Button: "Clear Conversation"
  
- **Welcome Modal** (first launch only):
  - Headline: "Let's tune your model"
  - Subheading: "Tell me about your use case in plain English"
  - Input: Text area with placeholder
  - Button: "Get Recommendation"
  - Skip link: "Or browse models manually"

---

### Feature 2: Data Management (Upload & Dataset Discovery)

#### User Story
"As a non-technical user, I want to upload my CSV file and have the system automatically detect format issues and suggest fixes instead of manually reformatting data."

#### Functional Requirements

**2.1 Data Upload Interface**
- **UI Component**: Drag-and-drop zone + file browser
  ```
  ┌─────────────────────────────────────┐
  │ 📁 Upload Training Data              │
  │                                     │
  │  Drag files here or                 │
  │  [Browse Files]                     │
  │                                     │
  │  Supported: CSV, JSON, JSONL, Parquet │
  │  Max size: 500MB (per file)         │
  │  Recommended: 100-100K examples     │
  └─────────────────────────────────────┘
  ```

**2.2 File Upload Flow**
1. User uploads CSV/JSON file
2. System reads first 100 rows
3. Auto-detect schema:
   ```python
   # Pseudo-code for auto-detection
   columns = detect_columns(file)
   data_types = infer_types(columns)
   potential_input_col = find_most_likely_input(columns)
   potential_output_col = find_most_likely_output(columns)
   ```
4. Display detected schema to user:
   ```
   ✅ Column Detected: "customer_message" → Input Text
   ✅ Column Detected: "support_reply" → Output Text
   ⚠️ Column: "timestamp" → Not needed for training
   ⚠️ Column: "user_id" → Not needed for training
   
   [Confirm & Continue] [Adjust Mapping]
   ```
5. If confidence < 60%, show manual mapping UI:
   ```
   Which column should be the INPUT (user message)?
   [Dropdown: customer_message | support_reply | timestamp]
   
   Which column should be the OUTPUT (model response)?
   [Dropdown: customer_message | support_reply | timestamp]
   ```

**2.3 Data Quality Analysis**
After upload, scan dataset for issues:

| Issue | Detection | Action |
|-------|-----------|--------|
| **Missing Values** | Rows with null/empty input or output | Show % affected, offer removal |
| **Duplicates** | Exact duplicate row pairs | Show count, offer dedup |
| **Length Imbalance** | Input/output ratio extreme (e.g., 10:1) | Flag, suggest review |
| **Text Encoding** | UTF-8 errors, special chars | Auto-fix or warn |
| **Dataset Size** | Too small (<100) or very large (>100K) | Warn about overfitting/cost |
| **Language Mismatch** | Detected != user's expected language | Ask to confirm |

**Example UI**:
```
📊 Data Quality Report
━━━━━━━━━━━━━━━━━━━━━
✅ Total Examples: 1,247
✅ Language: English (detected)
⚠️ Missing Values: 23 rows (1.8%) → [Auto-remove]
⚠️ Duplicates: 12 pairs found → [Remove Duplicates]
✅ Text Encoding: Valid UTF-8
✅ Average Lengths: Input 45 tokens, Output 78 tokens
✅ Split Recommendation: 80% train (998), 20% test (249)

[Looks Good!] [Review Examples] [Fix Issues]
```

**2.4 Data Preview**
- Interactive table showing first 10 rows:
  ```
  # | Input (customer_message) | Output (support_reply) | Quality Score
  1 | "How do I reset my pwd?" | "Go to Settings > Security > Reset Password" | ✅ 95%
  2 | "I can't login" | "Try clearing browser cache..." | ✅ 88%
  3 | "???" | "???" | ❌ 12%
  ```
- Sortable/filterable by quality score
- Hide/show specific examples
- "Delete this example" for clearly bad rows

**2.5 Alternative: Dataset Discovery (Brainstorm Agent Recommendation)**

If user has no data, offer:
```
No training data? I can help find datasets!

[Find Open-Source Datasets]

"What type of data are you looking for?"
- Customer support conversations (10K examples)
- Technical documentation Q&A (50K examples)
- Product review classification (100K examples)
- Medical Q&A (20K examples)
- [Search for custom dataset]

I'll download, verify, and format it for you.
Estimated size: ~500MB
Ready? [Proceed] [Upload My Own]
```

**Backend for Dataset Discovery**:
- Integration with Hugging Face Datasets
- Kaggle API for popular datasets
- Wikipedia download for general knowledge
- Query logic: Match user's use case → rank by quality/size → provide download link

**2.6 Data Storage**
- **Local Storage** (MVP): Store in `~/.vibeflow/projects/{project-id}/data/`
- **Cloud Option** (v1.1): Integrate with Hugging Face Hub for managed storage
- **Versioning**: Each uploaded dataset gets unique ID + timestamp
- **Access Control**: Project-level (user owns all data)

---

### Feature 3: Model Configuration & Training Setup

#### User Story
"As a developer who doesn't understand LoRA vs QLoRA, I want the system to configure training parameters automatically so I can start training immediately, but I also want to see what's happening."

#### Functional Requirements

**3.1 Training Configuration Panel**

**Auto-Config Flow** (default, for non-experts):
```
┌─────────────────────────────────────────┐
│ Training Configuration                  │
├─────────────────────────────────────────┤
│                                         │
│ 🤖 AI Recommended Setup                 │
│                                         │
│ Base Model: Mistral-7B                 │
│ Method: LoRA                            │
│ Rank: 8 (lora_r)                       │
│ Learning Rate: 5e-4                     │
│ Batch Size: 4                           │
│ Epochs: 3                               │
│ Estimated Cost: $18                     │
│ Estimated Time: 2-4 hours               │
│                                         │
│ [Why These Settings?] [Customize]       │
│                                         │
│ [Start Training]                        │
└─────────────────────────────────────────┘
```

**3.2 "Why These Settings?" Explainer**
Click → Reveals explanation:
```
🎯 Method: Why LoRA?
You have 1,247 training examples and 1 A100 GPU.
LoRA balances efficiency (4% trainable params) and quality 
(better than QLoRA for your dataset size).

For reference:
- Full Fine-tuning: Better quality, need 80GB VRAM
- QLoRA: Great for small GPUs, slight quality tradeoff
- LoRA: Best balance for your setup ← Recommended

📊 Hyperparameters: Why These Values?
Rank=8: Tested on 1K-10K examples; good generalization
LR=5e-4: Standard for LoRA on customer support tasks
Batch=4: Fits in A100 memory; good gradient updates
Epochs=3: Prevents overfitting on your dataset size

Want to adjust? See "Advanced Mode" below.
```

**3.3 Advanced Configuration (Optional)**
Toggle: "Show Advanced Settings" → Reveal:
```
┌─────────────────────────────────────────┐
│ Advanced Training Configuration          │
├─────────────────────────────────────────┤
│                                         │
│ Method: [LoRA ▼] / Full / QLoRA        │
│                                         │
│ LoRA Settings:                          │
│ - LoRA Rank (r): [8    ]  ?            │
│ - LoRA Alpha: [16   ]  ?                │
│ - Target Modules: [q_proj, v_proj] ?   │
│                                         │
│ Training Settings:                      │
│ - Learning Rate: [5e-4] ?              │
│ - Batch Size: [4    ] ?                 │
│ - Epochs: [3    ] ?                     │
│ - Gradient Accumulation: [2    ] ?     │
│ - Warmup Steps: [100] ?                │
│ - Weight Decay: [0.01] ?               │
│ - Optimizer: [AdamW ▼] ?               │
│                                         │
│ [Reset to Recommended] [Save Config]   │
│                                         │
│ Real-time Cost Estimate: $18.50        │
│ Real-time Time Estimate: 3h 22m        │
└─────────────────────────────────────────┘
```

Each `?` icon links to:
```
LoRA Rank (r):
- Controls how much you adapt the model
- Higher = more capacity but slower & more GPU memory
- Typical range: 4-64
- Your case: 8 recommended for 1K-10K examples
- Docs: [Link to LoRA paper]
```

**3.4 Cost & Time Estimation**

**Real-Time Calculation**:
```javascript
// Pseudo-code
function estimateCost(model, method, datasetSize, hyperparams) {
  baseCostPerHour = {
    'Mistral-7B': 0.50,  // Cost per GPU hour
    'LLaMA-13B': 0.75,
    'LLaMA-70B': 3.00
  }
  
  // Adjust for method
  if (method === 'full') efficiency = 1.0
  if (method === 'lora') efficiency = 0.3
  if (method === 'qlora') efficiency = 0.1
  
  // Estimate training time
  examplesPerSecond = (method === 'full') ? 10 : 50
  estimatedSeconds = datasetSize / examplesPerSecond
  estimatedHours = estimatedSeconds / 3600
  
  cost = estimatedHours * baseCostPerHour[model] * efficiency
  
  return {
    cost: cost.toFixed(2),
    time: formatTime(estimatedHours),
    costPerExample: (cost / datasetSize).toFixed(4)
  }
}
```

**Display Format**:
```
💰 Cost Estimate
   Training: $18.50 (actual GPU usage)
   Storage: ~$2/month (model weights)
   Total First Month: $20.50

⏱️ Time Estimate
   Training: 2-4 hours (actual time)
   Queue Time: <5 min (if busy)
   Total: 2-4 hours

💡 Tips to Reduce Cost:
   - Reduce epochs from 3 to 2 → saves $6
   - Use QLoRA instead of LoRA → saves $12
   - Use smaller batch size → faster but less stable
```

**3.5 Hardware Selection**
- **Auto-select** based on model + method:
  ```
  Recommended Hardware: NVIDIA A100 (40GB)
  Reason: Best for Mistral-7B + LoRA training
  
  Alternative Options:
  ✅ NVIDIA L40S (48GB) - $0.80/hr, 2-3% slower
  ⚠️ NVIDIA RTX4090 (24GB) - $0.50/hr, requires QLoRA
  ❌ T4 - Doesn't fit this model+method combo
  
  [Use Recommended] [Select Alternative]
  ```
- **Backend**: Query RunPod/Lambda Labs APIs for:
  - Available GPUs
  - Current pricing
  - Queue time
  - Estimated availability

**3.6 Pre-Training Checklist**
Before clicking "Start Training", verify:
```
✅ Training Data: 1,247 examples loaded
✅ Test Data: 249 examples reserved
✅ Base Model: Mistral-7B selected
✅ Method: LoRA configured
✅ Hardware: A100 requested
✅ Estimated Cost: $18.50 (within budget)
✅ Estimated Time: 2-4 hours

Everything looks good!
[Start Training] or [Back to Review]
```

---

### Feature 4: Real-Time Training Monitoring

#### User Story
"While my model trains, I want to see live progress with loss curves, resource usage, and ability to stop training if something looks wrong."

#### Functional Requirements

**4.1 Training Dashboard**

**Main View** (while training):
```
┌─────────────────────────────────────────────────────┐
│ 🟢 Training In Progress (Model: v1 / Mistral-7B)   │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Epoch 2/3 ████████░░ 67% | Steps 450/675           │
│ Time Elapsed: 1h 22m | Remaining: ~45m             │
│                                                     │
│ ┌──────────────────────────────────────┐           │
│ │ Loss Curve (Live)                    │           │
│ │                                      │           │
│ │       Loss↑                          │           │
│ │        4.2 ┌─                       │           │
│ │        4.0 │ \                      │           │
│ │        3.8 │  \___                  │           │
│ │        3.6 │      \___               │           │
│ │        3.4 │          ├─ Desired    │           │
│ │              ┴┴┴┴┴┴┴┴┴ Steps →    │           │
│ │           (decreasing = good!)      │           │
│ └──────────────────────────────────────┘           │
│                                                     │
│ 📊 Training Metrics                                │
│  • Current Loss: 3.42 ↘ (good trend)              │
│  • Learning Rate: 5e-4                            │
│  • GPU Memory: 38/40 GB (95%)                     │
│  • GPU Utilization: 92%                           │
│  • Tokens/sec: 850                                │
│                                                     │
│ 💾 Checkpoints Saved: 2                           │
│  • Checkpoint 1 (step 200): Loss 4.2              │
│  • Checkpoint 2 (step 450): Loss 3.42 ← Best      │
│                                                     │
│ 🔴 [Stop Training] | 📊 [View Logs]              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**4.2 Log Streaming**
- Real-time log output (hidden by default, clickable):
  ```
  [View Detailed Logs ▼]
  
  ─── Training Logs ──────────────────────────
  [14:32:15] Starting training...
  [14:32:16] Loaded model: Mistral-7B
  [14:32:22] Loaded 1247 training examples
  [14:32:23] LoRA config: rank=8, alpha=16
  [14:32:25] Initializing trainer...
  [14:32:30] Training started!
  [14:33:15] Epoch 1/3: Loss 4.234, LR 5e-4
  [14:33:45] Epoch 1/3: Loss 4.156, LR 5e-4
  [14:34:15] Epoch 1/3: Loss 4.089, LR 5e-4
  ...
  [Download Logs as .txt]
  ```

**4.3 Stop/Pause Training**
- **Stop Button**: Saves current checkpoint, gracefully terminates
  ```
  Are you sure? Stopping will:
  ✓ Save current model state
  ✓ Let you evaluate what trained so far
  ✓ Refund unused GPU time
  
  [Stop] [Cancel]
  ```
- **Pause Button** (v1.1): Pause training, resume later

**4.4 Alerts During Training**
- **GPU Memory High**: "⚠️ GPU memory at 95%. Model may crash. Reduce batch size."
- **Loss Not Decreasing**: "⚠️ Loss increasing for 5 steps. Consider lower learning rate."
- **Overfitting Risk**: "⚠️ Your dataset is small (1,247 examples). Consider reducing epochs to 2."

**4.5 Training Speed Indicators**
```
⚡ Training Efficiency
  Tokens/second: 850 (excellent)
  GPU Utilization: 92% (excellent)
  Data Loading: 95% (excellent)
  
  Your training is running optimally!
```

---

### Feature 5: Model Evaluation & Comparison

#### User Story
"After training completes, I want to see how my fine-tuned model compares to the base model using real examples and benchmarks, without needing to understand metrics like BLEU or ROUGE."

#### Functional Requirements

**5.1 Auto-Evaluation Pipeline**

**On Training Complete**, automatically run evaluation:
```
┌─────────────────────────────────────────────────────┐
│ 🎉 Training Complete! (Time: 2h 34m)               │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Running evaluation on test set...                   │
│ █████████████░░░░░░ 65%                            │
│                                                     │
│ Running benchmark tests...                         │
│ ██████░░░░░░░░░░░░░ 30%                            │
│                                                     │
│ Generating examples...                             │
│ █████████████████░░░ 85%                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Estimated wait**: 5-10 minutes

**5.2 Evaluation Results Dashboard**

**Quick Summary View**:
```
┌─────────────────────────────────────────────────────┐
│ Model Evaluation Results                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📈 Overall Score: 78/100 (Good)                    │
│    Base Model: 62/100 (Baseline)                   │
│    Improvement: +16 points ✅                       │
│                                                     │
│ Key Metrics:                                        │
│  • Accuracy on test set: 78% (vs 62%)             │
│  • Relevance: 82% (how relevant responses are)    │
│  • Fluency: 76% (how natural responses sound)     │
│  • Factuality: 74% (how accurate facts are)       │
│                                                     │
│ Quality Assessment:                                 │
│  ✅ Excellent: Shows domain understanding          │
│  ✅ Good: Follows instructions well                │
│  ⚠️ Fair: Sometimes verbose                        │
│  ❌ Poor: Occasionally repeats itself              │
│                                                     │
│ [Detailed Analysis] [Compare Examples]             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**5.3 Detailed Metrics Breakdown**

Click "Detailed Analysis":
```
┌─────────────────────────────────────────────────────┐
│ Detailed Performance Analysis                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Quantitative Metrics:                               │
│ ┌──────────────────────────────────────────┐       │
│ │ Metric        | Base Model | Fine-Tuned  │       │
│ ├──────────────────────────────────────────┤       │
│ │ Accuracy      | 62%        | 78% ↑ 16%  │       │
│ │ Precision     | 58%        | 75% ↑ 17%  │       │
│ │ Recall        | 65%        | 81% ↑ 16%  │       │
│ │ F1-Score      | 61%        | 78% ↑ 17%  │       │
│ │ Avg Response  | 156 tokens | 142 tokens │       │
│ │ Length        | (verbose)  | (concise)   │       │
│ └──────────────────────────────────────────┘       │
│                                                     │
│ Qualitative Feedback (LLM-as-Judge):               │
│  • Coherence: ⭐⭐⭐⭐⭐ (5/5) vs ⭐⭐⭐⭐ (4/5)    │
│  • Helpfulness: ⭐⭐⭐⭐⭐ (5/5) vs ⭐⭐⭐ (3/5)     │
│  • Safety: ⭐⭐⭐⭐⭐ (5/5) vs ⭐⭐⭐⭐⭐ (5/5)      │
│                                                     │
│ Error Analysis:                                     │
│  Base model errors: 38 instances                    │
│  Fine-tuned errors: 22 instances                    │
│  Improvement: 42% fewer errors ✅                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**5.4 Side-by-Side Example Comparison**

Click "Compare Examples":
```
┌─────────────────────────────────────────────────────┐
│ Example Comparison (Test Set)                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Example 1: Customer asks "How to reset password?"  │
│                                                     │
│ Input: "How do I reset my password?"               │
│                                                     │
│ 📊 Base Model (Mistral-7B):                        │
│ "To reset your password, go to the login page     │
│  and click 'Forgot Password'. You'll receive      │
│  an email with instructions. If you don't see     │
│  the email, check your spam folder. You can also  │
│  contact support at support@company.com or call   │
│  1-800-555-0123."                                 │
│ Quality: ⭐⭐⭐ (Good but verbose)                  │
│                                                     │
│ ✅ Fine-Tuned Model (v1):                         │
│ "Go to Login > Forgot Password. Check your email  │
│  for reset instructions. If no email, contact     │
│  support@company.com."                            │
│ Quality: ⭐⭐⭐⭐⭐ (Concise & accurate) ✅         │
│                                                     │
│ Feedback: Fine-tuned model learned customer       │
│ support style: direct, concise, helpful.          │
│                                                     │
│ [Next Example] [Previous] (1/10)                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**5.5 Problem Detection**

If model quality is low:
```
⚠️ Performance Below Expectations (64/100)

Possible reasons:
1. Training data quality: Some examples may have issues
   → [Review Data Quality]
   
2. Too few training examples (1,247 is on the small side)
   → [Find Similar Datasets] to augment
   
3. Hyperparameters might not be optimal
   → [Try Different Settings]
   
4. Model might be too small for this task
   → [Try Larger Model] (LLaMA-13B)

What do you want to try?
```

**5.6 Benchmark Testing** (Optional)

Offer standard benchmarks:
```
📚 Benchmark Tests (Optional)

Run your fine-tuned model against:

□ HellaSwag (commonsense reasoning) - 5 min
□ MMLU (general knowledge) - 10 min
□ TruthfulQA (factual accuracy) - 5 min
□ Custom Benchmark (your own test set) - 2 min

Compare against:
- Base Mistral-7B
- Base LLaMA-7B
- Previous versions of your model

Run benchmarks? (This will use some GPU time)
[Run All] [Select] [Skip for Now]
```

---

### Feature 6: Model Deployment

#### User Story
"After validation, I want to deploy my model to production with a single click and get an API endpoint I can integrate into my app immediately."

#### Functional Requirements

**6.1 Deployment Options**

After evaluation, offer:
```
┌─────────────────────────────────────────────────────┐
│ Deploy Fine-Tuned Model                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Choose deployment method:                          │
│                                                     │
│ ☑️ REST API Endpoint (Recommended)                 │
│    - Accessible from anywhere                      │
│    - Scalable, pay per request                     │
│    - $0.001/request baseline                       │
│    - [Deploy]                                      │
│                                                     │
│ ○ Run Locally                                      │
│    - Download model weights (2-5 GB)              │
│    - Run on your own machine                       │
│    - No cloud costs                                │
│    - [Download]                                    │
│                                                     │
│ ○ Hugging Face Hub (v1.1)                         │
│    - Push to community model hub                   │
│    - Share with others                             │
│    - [Publish]                                     │
│                                                     │
│ ○ Cloud Deployment (v1.1)                         │
│    - AWS SageMaker / GCP Vertex AI                │
│    - Enterprise-grade scaling                      │
│    - [Connect Cloud Account]                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**6.2 REST API Endpoint Deployment**

Click "Deploy REST API":
```
┌─────────────────────────────────────────────────────┐
│ Creating API Endpoint...                            │
│ ████████████░░░░░░░░░ 60%                          │
│                                                     │
│ Steps:                                              │
│ ✅ Packaging model                                 │
│ ✅ Setting up vLLM server                          │
│ ⏳ Deploying container                             │
│ ○ Initializing API                                 │
│ ○ Running health checks                            │
│                                                     │
│ This may take 2-3 minutes...                       │
└─────────────────────────────────────────────────────┘
```

**Success Screen**:
```
┌─────────────────────────────────────────────────────┐
│ ✅ Your Model is Live!                             │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 🔗 API Endpoint:                                   │
│    https://api.vibeflow.ai/v1/models/mistral-v1   │
│    [Copy]                                          │
│                                                     │
│ 🔑 API Key:                                        │
│    sk_live_abcdef123456789...                      │
│    [Copy] [Regenerate] [Revoke]                    │
│                                                     │
│ 📚 Documentation:                                  │
│    curl -X POST https://api.vibeflow.ai/v1/complete \
│      -H "Authorization: Bearer sk_live_..." \
│      -H "Content-Type: application/json" \
│      -d '{"prompt": "Hello", "max_tokens": 100}'  │
│                                                     │
│    Response:                                        │
│    {                                               │
│      "id": "req_123abc",                          │
│      "completion": "Hello! How can I help...",    │
│      "tokens_used": 45,                           │
│      "cost": "$0.00045"                           │
│    }                                               │
│                                                     │
│ 🧪 Test Endpoint:                                 │
│    [Input] ________________  [Submit]             │
│    Response will appear here...                    │
│                                                     │
│ 📊 Usage Dashboard:                                │
│    • Requests Today: 0                             │
│    • Cost Today: $0.00                             │
│    • Uptime: 100%                                  │
│    [View Full Dashboard]                           │
│                                                     │
│ ⚙️ Settings:                                       │
│    • Rate Limit: 100 req/min                       │
│    • Timeout: 30 seconds                           │
│    • Model Replicas: 1                             │
│    [Advanced Options]                              │
│                                                     │
│ [Integrate into App] [Share] [Stop Model]          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**6.3 Local Download Option**

Click "Download":
```
┌─────────────────────────────────────────────────────┐
│ Download Your Model                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Model: Mistral-7B + LoRA Weights                   │
│ Size: 2.4 GB                                       │
│                                                     │
│ What you get:                                       │
│  • model.safetensors (model weights)              │
│  • adapter_config.json (LoRA config)              │
│  • adapter_model.bin (LoRA weights)               │
│  • requirements.txt (dependencies)                 │
│  • inference_example.py (sample code)             │
│                                                     │
│ How to use locally:                                │
│  pip install -r requirements.txt                   │
│  python inference_example.py \                     │
│    --model_name mistralai/Mistral-7B \           │
│    --adapter_path ./adapter_config.json            │
│                                                     │
│ [Download] (~5 minutes)                            │
│                                                     │
│ Alternative: Use with Ollama / LM Studio           │
│ [Instructions for Ollama]                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**6.4 Model Version Management**

After deployment:
```
📦 Deployed Models

Active: mistral-v1 (Fine-Tuned)
├─ Created: Nov 1, 2025 @ 2:45 PM
├─ Requests: 0 | Cost: $0.00
├─ Accuracy: 78/100
├─ Status: 🟢 Live
├─ Endpoint: https://api.vibeflow.ai/v1/models/mistral-v1
└─ Actions: [Test] [Logs] [Stop] [Rollback]

History:
○ mistral-base (Base Model)
├─ Created: Oct 31, 2025 @ 6:15 PM
├─ Requests: 0
├─ Accuracy: 62/100
├─ Status: ⚪ Inactive
└─ Actions: [Activate] [Delete]

[Train New Version]
```

**6.5 A/B Testing** (v1.1)

Split traffic between models:
```
A/B Test Configuration

Model A: mistral-v1 (Current) - 80%
Model B: mistral-v1-exp (Experimental) - 20%

Results after 1000 requests:
Model A Accuracy: 78% | Cost: $0.80
Model B Accuracy: 81% | Cost: $0.85

Model B performing 3% better!
[Switch to B] [Continue Test] [End Test]
```

---

### Feature 7: Project Management & History

#### Functional Requirements

**7.1 Project Sidebar**

Left sidebar shows project list:
```
📁 Projects

[+ New Project]

Recent:
📂 Customer Support Bot
   • Model: Mistral-7B v1
   • Status: 🟢 Deployed
   • Created: Nov 1, 2025

📂 Document Classifier
   • Model: LLaMA-7B v2
   • Status: 🟡 Training
   • Created: Oct 30, 2025

📂 Code Generator
   • Model: Phi-3 v1
   • Status: ⚪ Draft
   • Created: Oct 28, 2025
```

**7.2 Project Dashboard**

Click project → view full timeline:
```
┌─────────────────────────────────────────────────────┐
│ Project: Customer Support Bot                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📋 Project Info:                                   │
│  • Created: Nov 1, 2025                            │
│  • Total Training Time: 2h 34m                     │
│  • Total Spend: $24.50                             │
│  • Current Model: mistral-v1                       │
│                                                     │
│ 📊 Training History:                               │
│                                                     │
│ ✅ mistral-v1 (DEPLOYED)                           │
│    Trained: Nov 1, 2025 @ 2:45 PM                 │
│    Time: 2h 34m | Cost: $18.50                     │
│    Accuracy: 78% | Deployed @ 5:15 PM             │
│    [View Training Logs] [Download Model] [Compare] │
│                                                     │
│ ○ mistral-draft                                    │
│    Created: Oct 31, 2025 @ 8:00 PM                │
│    Status: Not trained                             │
│    Data: 1,247 examples                            │
│    [Resume Training] [Delete]                      │
│                                                     │
│ 📈 Model Comparison:                               │
│    [Compare All Versions]                          │
│                                                     │
│ 🔄 Version Control:                                │
│    mistral-v1 ← Current                            │
│    mistral-draft                                   │
│    [Git History] [Rollback]                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**7.3 Version Control Integration**

Git-style versioning for models:
```
commit a3f2d8c (HEAD -> main)
Author: User <user@example.com>
Date:   Nov 1, 2025 @ 2:45 PM

   Fine-tune Mistral-7B on customer support data
   
   - Accuracy improved from 62% to 78%
   - Using LoRA method (4% trainable params)
   - Cost: $18.50 for training

commit 7e1c9a2 (base)
Author: System <system@vibeflow.ai>
Date:   Nov 1, 2025 @ 8:00 AM

   Created project: Customer Support Bot
   - Base Model: Mistral-7B
   - Data: 1,247 examples uploaded

[View Full History] [Create Branch] [Merge]
```

---

## Technical Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Machine                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  VS Code Fork with VibeFlow Extension                │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ Frontend (React/TypeScript)                      │ │ │
│  │  │ - Brainstorm Agent Chat UI                      │ │ │
│  │  │ - Data Upload/Management                        │ │ │
│  │  │ - Training Config Panel                         │ │ │
│  │  │ - Monitoring Dashboard                          │ │ │
│  │  │ - Evaluation Results                            │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │                       ↓                               │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ Local State (Electron IPC)                       │ │ │
│  │  │ - Project metadata                              │ │ │
│  │  │ - Dataset cache                                 │ │ │
│  │  │ - Configuration profiles                        │ │ │
│  │  │ - Git history                                   │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ↓                           ↓
┌──────────────────────┐   ┌──────────────────────┐
│   VibeFlow Backend   │   │    Remote GPU (*)    │
│   (Python FastAPI)   │   │  RunPod / Lambda     │
│                      │   │                      │
│ • Auth & Sessions    │   │ • Training Jobs      │
│ • Project Storage    │   │ • Inference Server   │
│ • Evaluation Engine  │   │ • Model Weights      │
│ • Cost Tracking      │   │ • vLLM Server        │
│ • API Gateway        │   │                      │
│ • Monitoring         │   │ (*) Zero-cost MVP:   │
│                      │   │ Just instructions    │
└──────────────────────┘   │ for RunPod CLI       │
         ↓                  └──────────────────────┘
  ┌────────────────────┐
  │  Backend Services  │
  │                    │
  │ • Hugging Face API │
  │ • Gemini 2.5 API   │
  │ • Kaggle API       │
  │ • Database (Local) │
  └────────────────────┘
```

### Component Details

#### Frontend Architecture
- **Base**: VS Code Fork (open-source)
- **Extension System**: Leverage VS Code's extension API
- **UI Components**: React-based panels
- **State Management**: Redux or Zustand
- **Build**: Webpack (VS Code standard)

#### Backend Architecture
- **Framework**: Python FastAPI
- **Server**: Uvicorn
- **Task Queue**: Celery + Redis (for async jobs)
- **Database**: SQLite (MVP) → PostgreSQL (production)
- **Storage**: Local filesystem (MVP) → S3 (production)
- **API Style**: RESTful with WebSocket for real-time updates

#### GPU/Training Infrastructure
- **MVP Strategy**: Use external services' CLIs
  - RunPod CLI for training
  - Lambda Labs API for GPU provisioning
  - vLLM as-is service
- **No infrastructure to manage in MVP**
- **v1.0**: Self-hosted option

#### Security & Authentication
- **Auth**: API keys for users (stored hashed)
- **Encryption**: TLS for all API communication
- **Data Privacy**: All user data stored locally by default
- **Model Weights**: Users own all weights; optional cloud backup

---

## UI/UX Specifications

### Design System

#### Color Palette
```
Primary: #3B82F6 (Blue - actions, highlights)
Secondary: #8B5CF6 (Purple - success, completion)
Danger: #EF4444 (Red - errors, stop)
Warning: #F59E0B (Amber - warnings, caution)
Success: #10B981 (Green - completion)
Neutral: #F3F4F6 → #6B7280 (Light to dark grays)
```

#### Typography
- **Headings**: Inter / Segoe UI, 600-700 weight
- **Body**: Inter / Segoe UI, 400 weight, 14px
- **Monospace**: Fira Code / Monaco (for code/logs)
- **Line Height**: 1.5 for readability

#### Spacing
- **Base unit**: 8px
- **Gutters**: 16px, 24px, 32px
- **Component padding**: 12px-16px

#### Components
- **Buttons**: Solid, Ghost, Outlined variants
- **Inputs**: Text, Textarea, Select, Checkbox, Radio
- **Cards**: Subtle shadows, 4px border-radius
- **Modals**: Centered, dimmed overlay
- **Tooltips**: Dark background, light text, 0.2s fade

### Page/Panel Layouts

#### Layout 1: Welcome/Onboarding
```
┌─────────────────────────────────────────┐
│  VibeFlow Logo                          │
│  "Fine-Tune LLMs Like You Code"         │
│                                         │
│  [Get Started] [Browse Examples]        │
│                                         │
│  OR                                     │
│                                         │
│  3-Step Overview Video (60s)            │
│                                         │
│  "Start with Brainstorm Agent" →        │
└─────────────────────────────────────────┘
```

#### Layout 2: Main IDE (3-Column)
```
┌──────┬────────────────────┬─────────────┐
│      │                    │             │
│Sidebar│  Main Editor Area │ Right Panel │
│      │                    │             │
│ • Projects      │ Brainstorm Chat    │ Properties
│ • Settings      │ Data Upload        │ Config
│ • Help          │ Training Monitor   │ Metrics
│ • Docs          │ Evaluation Results │ Deploy
│      │                    │             │
└──────┴────────────────────┴─────────────┘
```

#### Layout 3: Full-Width Dashboard (Training)
```
┌─────────────────────────────────────────┐
│ Training Dashboard (Mistral-7B, Epoch 2/3)
│                                         │
│ Progress Bar + Time Estimate            │
│ Live Loss Curve Graph                   │
│ Resource Monitoring (GPU, Memory)       │
│ Checkpoint Save History                 │
│ [Stop] [Logs] [Settings]               │
│                                         │
└─────────────────────────────────────────┘
```

### Responsive Design
- **Desktop**: Full 3-column layout (1200px+)
- **Tablet**: 2-column (sidebar + main, 768px+)
- **Mobile**: Single column (all stacked, <768px)
- **Minimum width**: 320px

### Accessibility
- **Color Contrast**: WCAG AA compliant
- **Keyboard Navigation**: Tab through all interactive elements
- **Screen Reader**: ARIA labels on all components
- **Focus States**: Visible 2px outline

---

## Data Management & Workflow

### Data Storage (MVP)

**Local Storage Structure**:
```
~/.vibeflow/
├── projects/
│   ├── project-1-uuid/
│   │   ├── config.json (project metadata)
│   │   ├── data/
│   │   │   ├── raw_upload.csv
│   │   │   ├── processed.jsonl (formatted for training)
│   │   │   ├── train.jsonl (80%)
│   │   │   └── test.jsonl (20%)
│   │   ├── models/
│   │   │   ├── base-mistral-7b/
│   │   │   └── lora-weights-v1/
│   │   ├── checkpoints/
│   │   │   ├── epoch-1/
│   │   │   └── epoch-2/
│   │   ├── evaluation/
│   │   │   ├── metrics.json
│   │   │   ├── examples-comparison.json
│   │   │   └── benchmark-results.json
│   │   └── logs/
│   │       ├── training.log
│   │       └── evaluation.log
│   └── project-2-uuid/
├── cache/
│   ├── huggingface-models/ (downloaded models)
│   └── datasets/ (cached datasets)
└── config.json (user settings, API keys)
```

### Data Flow (Complete Workflow)

```
Step 1: USER INPUT
┌──────────────────────┐
│ User describes goal  │
│ in Brainstorm Chat   │
└──────┬───────────────┘
       ↓
Step 2: BRAINSTORM AGENT
┌──────────────────────────────────────┐
│ AI recommends:                       │
│ - Model: Mistral-7B                  │
│ - Method: LoRA                       │
│ - Hyperparams                        │
│ - Cost/Time estimate                 │
└──────┬───────────────────────────────┘
       ↓
Step 3: DATA UPLOAD
┌──────────────────────────────────────┐
│ User uploads CSV/JSON                │
│ System detects schema                │
│ Quality analysis runs                │
│ Train/test split (80/20)             │
└──────┬───────────────────────────────┘
       ↓
Step 4: CONFIG REVIEW
┌──────────────────────────────────────┐
│ User sees recommended settings       │
│ Can customize if desired             │
│ Final cost/time estimate shown       │
└──────┬───────────────────────────────┘
       ↓
Step 5: TRAINING
┌──────────────────────────────────────┐
│ Model downloaded (if needed)         │
│ GPU provisioned (RunPod/Lambda)      │
│ LoRA training runs                   │
│ Real-time monitoring shown           │
│ Checkpoints saved                    │
└──────┬───────────────────────────────┘
       ↓
Step 6: EVALUATION
┌──────────────────────────────────────┐
│ Test set evaluation runs             │
│ Metrics calculated                   │
│ Example comparison generated         │
│ Quality score computed               │
└──────┬───────────────────────────────┘
       ↓
Step 7: DEPLOYMENT
┌──────────────────────────────────────┐
│ User chooses deployment option:      │
│ a) REST API endpoint                 │
│ b) Download locally                  │
│ c) Publish to Hub (v1.1)             │
│ API endpoint live in <2 min          │
└──────┬───────────────────────────────┘
       ↓
Step 8: MONITORING
┌──────────────────────────────────────┐
│ User sees usage stats                │
│ Cost tracking                        │
│ Performance monitoring               │
│ Can iterate/retrain                  │
└──────────────────────────────────────┘
```

### Dataset Format Support

**Accepted Formats** (on upload):
- **CSV**: Standard comma/tab-separated
- **JSON**: Array of objects or newline-delimited
- **JSONL**: One JSON object per line (streaming)
- **Parquet**: Binary columnar format

**Required Columns**:
Minimum: `input` + `output` (user-friendly names OK)
```
Examples of acceptable column names:
- "instruction" / "response"
- "question" / "answer"
- "user_message" / "assistant_message"
- "prompt" / "completion"
- "text_input" / "text_output"
```

**Auto-Detection Logic**:
```python
# Pseudo-code
def detect_columns(df):
    possible_input = ["instruction", "question", "prompt", 
                     "user_message", "text_input", "input"]
    possible_output = ["response", "answer", "completion", 
                      "assistant_message", "text_output", "output"]
    
    input_col = first(col for col in df.columns 
                     if col.lower() in possible_input)
    output_col = first(col for col in df.columns 
                      if col.lower() in possible_output)
    
    if not input_col or not output_col:
        confidence = 0.3  # Low; ask user
    else:
        confidence = 0.95  # High; proceed
    
    return {
        input: input_col,
        output: output_col,
        confidence: confidence
    }
```

### Data Quality Thresholds

| Issue | Threshold | Action |
|-------|-----------|--------|
| **Missing Values** | >5% | Warn, offer auto-remove |
| **Duplicates** | >2% | Warn, offer dedup |
| **Avg Input Length** | <10 tokens | Warn (too short) |
| **Avg Output Length** | >1000 tokens | Warn (too long) |
| **Input/Output Imbalance** | >10:1 ratio | Flag, suggest review |
| **Dataset Size** | <100 examples | Warn (overfitting risk) |
| **Dataset Size** | >1M examples | Info (large cost) |
| **Language Detection** | Mismatch | Ask confirmation |

---

## Fine-Tuning Engine

### Training Pipeline

**Architecture**:
```
Input (Training Data)
    ↓
[Data Loader] (HuggingFace Datasets)
    ↓
[Tokenizer] (Model-specific)
    ↓
[LoRA/QLoRA Setup] (PEFT library)
    ↓
[Trainer] (HuggingFace Transformers)
    ↓
[Checkpoint Manager] (Save every N steps)
    ↓
Output (Fine-tuned weights)
```

### Training Configurations

#### Default Configurations by Use Case

**Use Case 1: Customer Support (Recommended for 1K-10K examples)**
```yaml
model_name: "mistralai/Mistral-7B"
method: "lora"
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj"]
learning_rate: 5e-4
num_epochs: 3
per_device_batch_size: 4
gradient_accumulation_steps: 2
warmup_steps: 100
max_steps: 500
logging_steps: 50
eval_steps: 100
save_steps: 100
save_total_limit: 3
weight_decay: 0.01
optimizer: "adamw_8bit"
fp16: true
gradient_checkpointing: true
```

**Use Case 2: Technical Q&A (1K-10K examples)**
```yaml
model_name: "mistralai/Mistral-7B"
method: "lora"
lora_r: 16  # Slightly higher for technical domain
lora_alpha: 32
learning_rate: 2e-4  # Lower LR for stability
num_epochs: 5  # More epochs for domain adaptation
per_device_batch_size: 2  # Smaller batch for memory
gradient_accumulation_steps: 4
warmup_steps: 200
```

**Use Case 3: Code Generation (10K-100K examples)**
```yaml
model_name: "meta-llama/Llama-2-7b"
method: "lora"
lora_r: 32  # Higher rank for code complexity
learning_rate: 1e-4
num_epochs: 2  # Fewer epochs to avoid overfitting with large data
per_device_batch_size: 8
gradient_accumulation_steps: 1
```

### Supported Methods

#### Method 1: LoRA (Default)
- **Trainable Parameters**: ~4% of full model
- **Memory Requirement**: ~2x model size
- **Speed**: 3-5x faster than full fine-tuning
- **Quality**: Near-full fine-tuning quality
- **Recommended for**: Most users, most cases
- **Implementation**: HuggingFace PEFT

#### Method 2: QLoRA (For Resource-Constrained)
- **Trainable Parameters**: ~4% of full model
- **Memory Requirement**: ~0.5x model size (4-bit quantization)
- **Speed**: Slightly slower than LoRA
- **Quality**: 95% of LoRA quality, acceptable for most
- **Recommended for**: Single consumer GPU (RTX4090, etc.)
- **Implementation**: HuggingFace PEFT + bitsandbytes

#### Method 3: Full Fine-Tuning (For Maximum Quality)
- **Trainable Parameters**: 100% of model
- **Memory Requirement**: ~8x model size
- **Speed**: Slow (1x reference)
- **Quality**: Maximum possible
- **Recommended for**: Large datasets (>100K), enterprise users
- **Implementation**: Standard HuggingFace Transformers

### Hyperparameter Recommendations

**Learning Rate Selection**:
```
Small dataset (100-1K):      5e-4 (LoRA), 1e-4 (Full)
Medium dataset (1K-10K):     5e-4 (LoRA), 5e-5 (Full)
Large dataset (10K-100K):    2e-4 (LoRA), 2e-5 (Full)
Very large (100K+):          1e-4 (LoRA), 1e-5 (Full)
```

**Batch Size Selection**:
```
LoRA on A100 (40GB):    8-16
LoRA on A100 (80GB):    32
LoRA on L40S (48GB):    12
LoRA on RTX4090 (24GB): 2-4 (consider gradient accumulation)
QLoRA on RTX4090:       4-8
```

**Epoch Selection**:
```
Small dataset (<1K):     5-10 epochs (more iterations needed)
Medium (1K-10K):         3-5 epochs (sweet spot)
Large (10K+):            1-2 epochs (avoid overfitting)
```

### Cost Estimation Algorithm

```python
def estimate_cost(model, method, data_size, hardware):
    # Base hourly costs (subject to change)
    gpu_hourly_cost = {
        'A100-40GB': 0.50,
        'A100-80GB': 0.75,
        'L40S': 0.60,
        'RTX4090': 0.30,
        'T4': 0.15
    }
    
    # Training time estimation
    tokens_per_second = {
        'LoRA': 800,
        'QLoRA': 400,
        'Full': 200
    }
    
    total_tokens = data_size * avg_tokens_per_example  # ~100
    estimated_seconds = total_tokens / tokens_per_second[method]
    estimated_hours = estimated_seconds / 3600
    
    # Add overhead
    estimated_hours *= 1.2  # 20% overhead for data loading, etc.
    
    total_cost = estimated_hours * gpu_hourly_cost[hardware]
    
    return {
        'estimated_cost': total_cost,
        'estimated_hours': estimated_hours,
        'cost_per_example': total_cost / data_size,
        'confidence': 0.85  # ±15% variance expected
    }
```

### Error Handling During Training

| Error | Cause | Solution |
|-------|-------|----------|
| **CUDA Out of Memory** | Batch size too large | Auto-reduce batch size, suggest QLoRA |
| **Loss Not Decreasing** | Learning rate too high | Suggest reduce by half, continue |
| **Diverging Loss** | Learning rate too high | Strongly suggest reduce by 10x |
| **Very Slow Training** | GPU memory swapping | Suggest smaller batch size |

---

## Evaluation & Deployment

### Evaluation Metrics

#### Automatic Metrics (Run on Test Set)

**1. Exact Match Accuracy**
```
Measures: % of predictions exactly matching reference output
Use case: Classification tasks, Q&A with single answers
Formula: (correct_predictions / total_predictions) * 100
```

**2. Token-Level Metrics**
```
Precision: % of predicted tokens in reference
Recall: % of reference tokens in prediction
F1-Score: Harmonic mean of precision and recall
Use case: Any text generation task
Formula: Standard NLP metrics
```

**3. BLEU (Bilingual Evaluation Understudy)**
```
Measures: N-gram overlap between prediction and reference
Use case: Machine translation, paraphrasing
Range: 0-100 (higher is better)
Threshold: >25 = acceptable, >50 = good, >75 = excellent
```

**4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
```
Measures: Recall of important information
Use case: Summarization, abstractive tasks
Variants: ROUGE-L (longest common subsequence)
Range: 0-1 (higher is better)
Threshold: >0.3 = acceptable, >0.5 = good, >0.7 = excellent
```

**5. Length Penalty**
```
Measures: Average length of predictions vs. references
Use case: Detecting if model is too verbose
Calculation: actual_length / reference_length
Ideal: 0.9-1.1 (±10% deviation)
Alert if: >1.3 (too verbose) or <0.7 (too short)
```

**6. Perplexity**
```
Measures: How well model predicts next token
Use case: Language modeling baseline
Lower = better (good: <10, excellent: <5)
Not shown to users; internal reference
```

#### LLM-as-Judge Metrics (Using Gemini)

**System Prompt**:
```
You are an expert evaluator of AI responses.
Rate the following response on:
1. Relevance (1-5): Does it answer the question?
2. Coherence (1-5): Is it logically structured?
3. Factuality (1-5): Are claims accurate?
4. Completeness (1-5): Does it address all aspects?
5. Tone Match (1-5): Does it match expected tone?

Response:
[Test response here]

Reference (for context):
[Reference output here]

Provide scores as JSON:
{
  "relevance": 5,
  "coherence": 4,
  "factuality": 5,
  "completeness": 4,
  "tone_match": 5,
  "average": 4.6,
  "feedback": "Clear and helpful, very relevant"
}
```

#### Human-Readable Quality Score

```
Overall Quality Score = 
  (Accuracy * 0.4) + 
  (LLM_Judge_Score * 0.4) + 
  (Length_Appropriateness * 0.2)

Range: 0-100
- 80-100: Excellent (production ready)
- 60-79: Good (acceptable with review)
- 40-59: Fair (needs improvement)
- <40: Poor (retrain recommended)
```

### Evaluation Workflow

**On Training Complete**:

1. **Auto-Evaluation** (5-10 minutes)
   - Load fine-tuned model
   - Run on test set (249 examples)
   - Calculate automatic metrics
   - Run LLM-as-Judge on 20 random examples
   - Generate example comparisons

2. **Results Display**
   - Summary score (0-100)
   - Metric breakdown
   - Example comparisons
   - Recommendations (deploy, retrain, adjust)

3. **Quality Gates**
   - If score > 70: "Ready to Deploy"
   - If score 50-70: "Review Examples Before Deploy"
   - If score < 50: "Recommend Retraining"

---

## Analytics & Monitoring

### Usage Tracking

**Tracked Metrics**:
- Models trained (count)
- Training time (total hours)
- Data processed (total tokens)
- Cost (actual USD)
- Deployments (count)
- API calls (if deployed)
- Inference tokens (if deployed)

**User Dashboard**:
```
┌─────────────────────────────────────────┐
│ Your Usage (This Month)                 │
├─────────────────────────────────────────┤
│                                         │
│ Models Trained: 3                       │
│ Total Training Time: 8h 42m             │
│ Total Cost: $89.50                      │
│ Models Deployed: 2                      │
│ API Calls: 12,450                       │
│ Tokens Generated: 1.2M                  │
│                                         │
│ Billing Summary:                        │
│ Training: $89.50 (beta: free)           │
│ Deployment: $0 (first month free)       │
│ API Calls: $0 (free tier)               │
│ Total: $89.50                           │
│                                         │
│ Budget Alert: You've used 45% of        │
│ your $200 monthly budget                │
│ [Adjust Budget] [Upgrade Plan]          │
│                                         │
└─────────────────────────────────────────┘
```

### Model Performance Monitoring (Post-Deployment)

**Tracked Metrics**:
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx responses)
- Token usage (input/output)
- Cost per request
- Uptime (%)
- Quality drift (if feedback available)

**Alert Conditions**:
- Latency p95 > 5s
- Error rate > 1%
- Cost per request > expected
- Uptime < 99.9%

---

## Security & Privacy

### Data Security

**Encryption**:
- At rest: AES-256 for stored data
- In transit: TLS 1.3 for all API communications
- Keys: Managed by OS keychain / credential manager

**Data Retention**:
- User models: Retained indefinitely (user-owned)
- Training logs: 90 days retention
- Evaluation results: Retained indefinitely
- Usage analytics: Anonymized, 1 year retention

### User Authentication

**MVP**:
- Local authentication (username + password)
- Hashed with bcrypt
- Session tokens (JWT)
- Expiry: 30 days

**v1.0**:
- OAuth2 integration (GitHub, Google)
- 2FA support

### Model Privacy

**User Data Handling**:
- All training data stored locally by default
- No data sent to VibeFlow servers unless user explicitly uploads
- VibeFlow never trains on user data
- User owns all fine-tuned weights

**Telemetry**:
- Optional: Telemetry (disabled by default)
- If enabled: Error logs, usage patterns (no data content)
- Users can disable anytime

### API Keys

**Generation**:
- One key per project
- Rotatable
- Can be revoked instantly

**Rate Limiting**:
- Free tier: 100 req/min per key
- Pro tier: 1000 req/min per key
- Enterprise: Custom limits

---

## Development Roadmap

### Phase 1: MVP (Weeks 1-8)

**Weeks 1-2: Foundation**
- [ ] Fork VS Code repository
- [ ] Set up build pipeline
- [ ] Create basic extension structure
- [ ] Implement Electron IPC

**Weeks 3-4: Core Features**
- [ ] Brainstorm Agent chat interface (with Gemini API)
- [ ] Data upload + format detection
- [ ] Training config panel (auto-config)
- [ ] Project management sidebar

**Weeks 5-6: Training Integration**
- [ ] RunPod API integration
- [ ] Training job submission
- [ ] Real-time monitoring dashboard
- [ ] Checkpoint management

**Weeks 7-8: Evaluation & Deployment**
- [ ] Auto-evaluation pipeline
- [ ] Results visualization
- [ ] API endpoint deployment
- [ ] Local model download

**MVP Deliverable**:
- VS Code with VibeFlow extension
- Complete end-to-end workflow
- Support for Mistral-7B only
- LoRA method only
- Local storage only

### Phase 2: Robustness (Weeks 9-12)

**Features**:
- [ ] Multiple model support (LLaMA, Phi-3)
- [ ] QLoRA method support
- [ ] Error recovery and retry logic
- [ ] Advanced config UI
- [ ] A/B testing framework

**Improvements**:
- [ ] Performance optimization
- [ ] UI/UX refinement based on beta feedback
- [ ] Documentation and tutorials
- [ ] Community Discord setup

**Beta Deliverable**:
- Production-ready MVP
- 10-15 paying users
- Clear product roadmap

### Phase 3: Scale (Weeks 13-16)

**Features**:
- [ ] Multi-model support (10+ models)
- [ ] Full fine-tuning method
- [ ] Dataset discovery (Hugging Face, Kaggle)
- [ ] Version control with Git
- [ ] Monitoring dashboard
- [ ] Usage analytics

**Deployment Options**:
- [ ] Self-hosted option
- [ ] Cloud deployment (AWS, GCP)
- [ ] Hugging Face Hub integration

**v1.0 Deliverable**:
- Publicly available product
- Enterprise-grade features
- Community-driven roadmap

---

## Success Metrics

### User Engagement

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Activation Rate** | 70%+ | % who complete first fine-tuning |
| **Retention Day 7** | 50%+ | % active 7 days after signup |
| **Retention Day 30** | 30%+ | % active 30 days after signup |
| **NPS Score** | 40+ | Net Promoter Score |

### Product Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to Deploy** | <4 hours | From data upload to live API |
| **Training Success Rate** | 95%+ | % of training jobs that complete |
| **Evaluation Accuracy** | >70% | Avg score of fine-tuned models |
| **API Uptime** | 99.9%+ | Deployment availability |

### Business Metrics

| Metric | Target | Timeline |
|--------|--------|----------|
| **Beta Users** | 10-15 | Week 8 |
| **Paying Users** | 5-8 | Week 8 |
| **MRR (Monthly Recurring Revenue)** | $2K-4K | Week 12 |
| **Churn Rate** | <5% | Week 16 |

---

## Open Source Stack

### Frontend

| Component | Library | Version | Reason |
|-----------|---------|---------|--------|
| IDE Base | VS Code | Latest | Battle-tested, familiar to devs |
| UI Framework | React | 18+ | VS Code extension standard |
| State Management | Zustand | Latest | Lightweight, Redux alternative |
| HTTP Client | Axios | Latest | Simple promise-based HTTP |
| Charts | Recharts | Latest | React charting, open source |
| Icons | Lucide React | Latest | Beautiful, free icon set |
| Styling | Tailwind CSS | Latest | Utility-first CSS framework |

### Backend

| Component | Library | Version | Reason |
|-----------|---------|---------|--------|
| Framework | FastAPI | 0.100+ | Modern, fast, async Python |
| Server | Uvicorn | Latest | ASGI server, production-ready |
| ORM | SQLAlchemy | 2.0+ | SQL toolkit, flexible |
| Task Queue | Celery | Latest | Distributed task processing |
| Message Broker | Redis | Latest | In-memory data store, job queue |
| Authentication | PyJWT | Latest | JWT token handling |
| Validation | Pydantic | 2.0+ | Data validation, serialization |

### ML/Training

| Component | Library | Version | Reason |
|-----------|---------|---------|--------|
| Transformers | HuggingFace | Latest | Standard LLM library |
| PEFT | HuggingFace | Latest | LoRA, QLoRA, adapters |
| Tokenizer | tiktoken | Latest | GPT tokenizer |
| Datasets | HuggingFace | Latest | Dataset loading/processing |
| Trainer | Transformers.Trainer | Latest | Training loop abstraction |
| Quantization | bitsandbytes | Latest | 4-bit quantization for QLoRA |
| Evaluation | evaluate | Latest | Standard metrics (BLEU, ROUGE) |

### Inference/Serving

| Component | Library | Version | Reason |
|-----------|---------|---------|--------|
| Inference Engine | vLLM | Latest | Fast LLM inference |
| Model Format | Safetensors | Latest | Safe model serialization |
| LLM Loading | llama-cpp-python | Latest | Lightweight inference |
| Quantization | GGUF Format | Latest | Efficient model distribution |

### Development Tools

| Component | Library | Version | Reason |
|-----------|---------|---------|--------|
| Build Tool | Webpack | 5+ | VS Code extension standard |
| Bundler | Vite | Latest | Fast build tool (alt: Webpack) |
| Package Manager | npm/pnpm | Latest | JavaScript package management |
| Testing | Jest | Latest | JavaScript testing framework |
| Linting | ESLint | Latest | Code quality |
| Type Checking | TypeScript | Latest | Type safety |
| Python Testing | pytest | Latest | Python testing framework |
| Code Formatting | Black | Latest | Python code formatter |

### External APIs (Free Tier Only for MVP)

| Service | API | Free Tier | Use |
|---------|-----|-----------|-----|
| Gemini 2.5 | Google AI | 60 req/day | Brainstorm Agent |
| Hugging Face | Models Hub | Public models | Download models |
| Datasets | Hugging Face | Public datasets | Dataset discovery |
| Kaggle | Datasets API | Public datasets | Dataset discovery |

### Infrastructure (No Cost MVP)

| Service | Alternative | Cost | Use |
|---------|-------------|------|-----|
| GPU Training | RunPod CLI | Pay-as-you-go | Model training |
| GPU Training | Lambda Labs | Pay-as-you-go | Model training |
| Storage | Local Filesystem | $0 | Model/data storage |
| Database | SQLite | $0 | Project metadata |

---

## Non-Functional Requirements

### Performance

- **Page Load Time**: <2 seconds
- **Training Config Update**: <200ms
- **Evaluation Results Load**: <3 seconds
- **API Response Time**: <200ms (p95)
- **Model Download**: <10 min (for 7B model)

### Scalability

- **Concurrent Users**: 100+ in MVP
- **Projects per User**: Unlimited
- **Models per Project**: Unlimited
- **Training Jobs**: 10 concurrent max (due to GPU cost)

### Reliability

- **Uptime**: 99.5% (MVP), 99.9% (v1.0)
- **Data Loss Risk**: <0.01%
- **Training Job Failure Recovery**: Auto-resume from last checkpoint
- **API Error Recovery**: Automatic retry with exponential backoff

### Compatibility

- **Operating Systems**: macOS, Linux, Windows
- **Browsers**: Not applicable (desktop app)
- **Node Versions**: 16+
- **Python Versions**: 3.9+
- **GPU Support**: NVIDIA CUDA 11.8+ (initially)

---

## Constraints & Assumptions

### Constraints

1. **No infrastructure spending** → Use external GPU services
2. **Open source only** → No proprietary code
3. **First version single-user** → No multi-user/team features in MVP
4. **Specific GPUs** → NVIDIA only (CUDA ecosystem)
5. **MVP scope** → 1 base model, 1 method, limited features

### Assumptions

1. Users have basic Python knowledge
2. Users have GPU access or willing to pay for it
3. Users have training data prepared
4. Users have internet connection for Gemini API
5. Target users are technical (developers, startups, ML eng)
6. LoRA/QLoRA satisfies 80% of MVP users' needs

---

## Glossary

| Term | Definition |
|------|-----------|
| **Fine-Tuning** | Updating all or part of a pre-trained LLM on new data |
| **LoRA** | Low-Rank Adaptation; efficient method training 4% of params |
| **QLoRA** | Quantized LoRA; enables training on small GPUs |
| **Brainstorm Agent** | AI assistant helping users make fine-tuning decisions |
| **PEFT** | Parameter-Efficient Fine-Tuning methods (LoRA, QLoRA, etc.) |
| **vLLM** | Fast inference engine for LLMs |
| **Tokenizer** | Converts text to model-understandable tokens |
| **Checkpoint** | Saved model state during training |
| **Evaluation** | Measuring fine-tuned model quality vs. base |
| **Deployment** | Making fine-tuned model available for use |
| **API Endpoint** | URL where deployed model can be called |
| **GPU** | Graphics Processing Unit (needed for training) |
| **VRAM** | Video RAM on GPU (limits model size) |

---

## Next Steps for Development

### Immediate (Week 1)
1. [ ] Create GitHub repository
2. [ ] Set up project structure
3. [ ] Fork VS Code
4. [ ] Document development setup

### Short-Term (Weeks 2-4)
1. [ ] Build UI mockups
2. [ ] Set up backend skeleton
3. [ ] Integrate Gemini API
4. [ ] Test with 3-5 internal users

### Medium-Term (Weeks 5-8)
1. [ ] Complete core features
2. [ ] End-to-end testing
3. [ ] Performance optimization
4. [ ] Beta user onboarding

### Long-Term (Weeks 9-16)
1. [ ] Feature refinement
2. [ ] Documentation
3. [ ] Community building
4. [ ] v1.0 preparation

---

## Contact & Support

For questions about this PRD, contact: [your-email@vibeflow.ai]

**Document Version Control**:
- v1.0 - Initial PRD - Nov 1, 2025
- v1.1 - [To be updated after user feedback]

---

**End of PRD**
