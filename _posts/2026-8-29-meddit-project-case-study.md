---

layout: post

title: "Meddit: Telehealth Platform"

date: 2026-08-27

tags: [Medical AI, NLP, Agentic AI, RAG, Explainable AI, Projects]

---

# Meddit

> An AI-powered telehealth platform that combines structured medical interviews, symptom-urgency assessment, RAG-powered medical question answering, and doctor-authored health content. The platform provides patients with personalized AI assistance alongside daily health tips, medical articles, and access to healthcare professionals. Its AI pipeline is built around specialized agents, where each LLM call has a defined role and prompt, while agent decisions, inputs, outputs, and reasoning context are tracked throughout the workflow to improve transparency, traceability, and explainability.

<!-- Hero image / project screenshot -->

![Meddit](/assets/img/meddit_hero.png)

## Overview

Meddit was developed as a **full telehealth platform** exploring how AI could assist patients throughout the early stages of seeking medical care, from understanding their symptoms to finding appropriate care.

At the center of the platform is an AI-driven **structured medical interview** that collects information about the patient's symptoms and medical history, assesses the **urgency of the situation**, and determines an appropriate next step. Rather than treating the interaction as a simple question-and-answer chatbot, Meddit maintains a structured representation of the ongoing interview and uses the information collected throughout the conversation to dynamically determine what should be asked next.

Beyond the medical interview, the platform provides **RAG-powered medical question answering**, allowing patients to ask general health-related questions and receive responses grounded in a controlled medical knowledge base. It also includes **doctor-authored medical articles and daily health tips**, giving patients access to health information outside of active AI consultations, as well as functionality for connecting patients with healthcare professionals when appropriate.

A key aspect of Meddit is its focus on **AI explainability and traceability**. Rather than treating the AI as a single black-box model, the system is organized around specialized agents, with each agent corresponding to an LLM call with a specific role and prompt. The inputs, outputs, and decisions of these agents are tracked throughout the workflow, making it possible to trace **which agent made a decision, what information it received, what it produced, and why that decision influenced the subsequent stage of the process**.

The platform is implemented as a complete application consisting of a **patient-facing mobile application, a doctor-facing interface, an administrative dashboard, and a FastAPI backend** responsible for authentication, medical interviews, AI processing, triage, retrieval, data management, and real-time communication between patients and healthcare professionals.

---
## The Problem

Meddit was developed in the context of a **Syrian healthcare system undergoing a fragile transition toward recovery** after years of conflict and underinvestment. The healthcare workforce has been significantly reduced by the migration of health professionals, while many healthcare facilities continue to operate with limited capacity. WHO has reported substantial gaps in the functionality of hospitals and primary healthcare centers, alongside continuing workforce shortages.

For many patients, this means that **accessing appropriate medical care can itself be a challenge**. Reaching a doctor may require travelling significant distances, waiting for an available appointment, or navigating a healthcare system with limited resources. These difficulties make it particularly important for patients to have support in understanding their symptoms, learning about their health, and determining **when and where to seek care**.

Even when medical services are available, patients may not know **how urgent their symptoms are** or **which type of medical specialist they should consult**. At the same time, limited access to reliable and understandable health information can make it difficult for people to recognize symptoms, understand basic health conditions, or make informed decisions about seeking care.

A patient might describe several symptoms in a single conversation, provide incomplete information, or mention an important detail only after discussing something else.

A conventional chatbot can respond to each message independently, but a medical interview requires something more:

* Symptoms need to be identified and tracked throughout the conversation.
* Important missing information needs to be requested.
* Previously provided medical history needs to be considered.
* New symptoms may change the relevance of subsequent questions.
* The overall combination of symptoms needs to be considered when assessing urgency.
* The system should distinguish between situations requiring general health guidance and situations requiring professional medical care.

Furthermore, when professional care is needed, **accessing a doctor should not necessarily require navigating the healthcare system alone**. Providing a way for patients to connect with doctors remotely can help reduce some of the barriers associated with distance, availability, and access to appropriate specialists.

This led us to explore whether AI could support patients through a **structured, context-aware medical interview** rather than a purely conversational chatbot, while also providing accessible health information and a pathway to **online consultations with doctors** when professional care is appropriate.

The goal was not to replace doctors, but to build a platform that could help patients **better understand their health, identify the appropriate next step, and reach professional care when necessary**.

---

## What We Built

Meddit combines an AI-powered medical interview with health information and access to professional care, delivered through a Flutter mobile app for patients, a web dashboard for doctors, and an asynchronous FastAPI backend.

At the core of the system is a multi-agent architecture coordinated by a central orchestrator (**ChatService**), which routes each patient message through a pipeline of specialized agents:

1. **Classify intent** : an intent classifier determines whether the message is a medical interview request, a general health question, an out-of-scope request, or something requiring clarification, and routes it accordingly.

2. **Clarify ambiguous input** : when the user's intent is unclear, a dedicated clarifier agent asks a short follow-up question before proceeding.

3. **Answer general health questions through RAG** : a retrieval-augmented generation pipeline enriches the query, retrieves relevant passages from a controlled, indexed medical knowledge base, reranks them for relevance, checks whether the retrieved context is sufficient, and only then generates an answer grounded in those sources, reducing hallucination and keeping responses traceable to their origin.

4. **Extract structured medical information** : a dedicated extraction agent converts free-form conversation into a structured medical record: chief complaint, symptoms and severity, vital signs, red-flag symptoms, medical history, lifestyle context, and pain assessment. The agent merges new input field-by-field into the existing record so that previously captured details are never lost, even across multiple turns.

5. **Assess urgency through a triage engine** : the structured medical information is evaluated against a five-level severity scale (Crisis, High, Medium, Low, None), or flagged as needing more information. Each triage decision is accompanied by an explicit rationale, the logic used to reach the classification and the logic used to exclude other paths, rather than a bare label.

6. **Recommend a medical specialty** : when the triage outcome indicates a doctor visit is appropriate, a recommendation agent matches the patient's condition against the platform's available specialties and returns a specific recommendation with a stated reason.

7. **Connect patients with doctors through online video consultations** : patients can browse available doctors filtered by the recommended specialty and book or join a live consultation directly from the app.

8. **Generate a structured SOAP report** : once a consultation is needed, the system automatically compiles the interview, triage assessment, and history into a standard SOAP-format report (Subjective, Objective, Assessment, Plan), giving the doctor a ready summary before the session starts.

9. **Monitor for safety and explainability** : every AI-generated output is paired with a computed confidence score, and all inputs/outputs are logged in an auditable trail, so that both the reasoning behind a triage decision and the sources behind a RAG answer can be reviewed and verified.

**Why this structure matters:** rather than treating the online consultation and health-education features as separate add-ons, they sit downstream of the same reasoning pipeline that drives the medical interview, so a patient's path naturally flows from *AI Medical Interview → Health Education (RAG) → Triage & Specialty Matching → Online Consultation → SOAP Summary for the Doctor*, with confidence scoring and monitoring running underneath the whole chain rather than bolted onto the end of it.

![application_flow](/assets/img/design.drawio.svg)
---
Here's a tightened, accurate rewrite aligned with the actual architecture described in your report:

---

## How It Works

### 1. Structured Medical Interview
The core of Meddit is an AI-driven medical interview, not a fixed questionnaire.

When a patient reports a symptom, the system determines what additional information is clinically relevant for example, if a patient mentions chest pain, it identifies severity, duration, location, and associated symptoms as the next things to clarify, rather than working through a scripted list.

The interview is **stateful**: instead of relying on raw conversation history, Meddit maintains a structured medical record (`MedicalInformationExtracted`) that persists across turns — chief complaint, symptoms, vital signs, red-flag symptoms, medical history, and pain assessment. This lets the system distinguish between what has already been established and what still needs to be collected, so the same question is never asked twice.

```json
{
  "medical_information_extracted": {
    "chief_complaint": "fever",
    "symptoms": [
      {
        "name": "fever",
        "severity_self_reported": null,
        "duration": null,
        "onset": "unknown"
      }
    ],
    "vital_signs": {
      "temperature_c": "high",
      "heart_rate": null,
      "blood_pressure": null,
      "respiratory_rate": null,
      "oxygen_saturation": null
    },
    "red_flag_symptoms": [],
    "medical_history": {
      "chronic_diseases": [],
      "current_medications": [],
      "allergies": []
    },
    "lifestyle_context": {
      "smoking": null,
      "pregnancy": null,
      "recent_surgery": null
    },
    "pain_assessment": {
      "pain_present": null,
      "pain_scale_0_10": null,
      "location": null,
      "type": null
    },
    "preliminary_flags": {
      "confusion": false,
      "loss_of_consciousness": false,
      "unresponsiveness": false,
      "focal_neurological_deficit": false,
      "severe_chest_pain": false,
      "bleeding": false,
      "cyanosis_or_pallor": false,
      "severe_respiratory_distress": false,
      "airway_compromise": false,
      "seizure_active_or_recent": false,
      "severe_trauma": false
    }
  }
}
```
---

### 2. Intent Classification
Not every patient message is a new medical complaint. A message might:

* Provide information about a symptom.
* Answer a previous question.
* Deny the presence of a symptom.
* Ask a general medical question.
* Provide additional medical history.
* Ask for clarification.
* Change or correct previously provided information.

Meddit's **intent classifier agent** determines which of these categories a message falls into — routing it to a medical interview, a general-information (RAG) path, a clarification step, or an out-of-scope response — before anything else happens. This routing happens first so that every downstream agent interprets the message in the context of the current interview state, rather than as an isolated input.

![Intent Classifier](/assets/img/intent_classifier.png)

---

### 3. Symptom Extraction
Once a message is classified as part of the medical interview, a dedicated **extraction agent** converts it into structured data: symptom name, presence or absence, self-reported severity, onset, and related context.

This matters because a symptom the patient explicitly denies is treated differently from one that simply hasn't come up yet — missing information is never silently interpreted as a negative finding. New information is merged field-by-field into the existing record, so a partial or incomplete update never overwrites data already collected earlier in the conversation.

---

### 4. Dynamic Interview State

The structured record becomes the current interview state, which the **triage agent** uses alongside the latest response to decide what's still missing and what the next step should be:

**Patient response → Intent classification → Symptom extraction → Interview state → Triage / next question → Patient response**

This loop lets the interview adapt to the patient rather than forcing everyone through the same sequence. If duration and severity have already been reported, the system moves on to the remaining clinically relevant gaps instead of re-asking. Once enough information is available, the state is handed to the triage engine to assess urgency (Crisis, High, Medium, Low, None, or More Info) and, when appropriate, to recommend a specialty and generate a SOAP summary for the doctor.

![meddit triage](/assets/img/meddit_triage.png)

---

## A Challenge We Encountered

LLMs are notoriously unreliable at updating nested JSON structures across multiple turns. Since the medical interview state is a fairly deep object — symptoms, vital signs, medical history, pain assessment, and more all nested together — asking a model to simply "update the record" each turn turned out to be fragile.

In practice, if a patient said *"I have a mild headache"* in turn one and *"It started yesterday"* in turn three, a standard extraction prompt would often regenerate the entire record from scratch based on the latest message alone. The duration would get added, but the severity or even the symptom name from an earlier turn could silently disappear — not because the patient retracted it, but because the model wasn't reliably carrying forward information it wasn't actively looking at. Over a long interview, this kind of silent overwrite could erase clinically important details without any visible error.

A related problem sat right next to this one: the system needed to tell the difference between a symptom the patient explicitly ruled out ("No chest pain") and a symptom that simply hadn't come up yet. Treating both cases as equivalent — or worse, letting the model guess — risked either fabricating negative findings or losing track of what still needed to be asked.

**How We Solved It**

We moved away from full-record regeneration and built an explicit **delta-merging pipeline** inside the Extraction Agent. Instead of asking the LLM to output the complete `MedicalInformationExtracted` object every turn, the agent only emits a partial update — the fields that changed based on the latest message. That delta is validated and then merged field-by-field into the persistent state vector, so information from earlier turns that isn't mentioned again is simply left untouched rather than being at risk of being overwritten.

To handle the negative-finding problem, we enforced a strict three-way distinction at the schema level: a field can be explicitly `false`/denied, explicitly populated with a value, or `null`. Null is never treated as a negative finding — it only ever means "not yet discussed" — which keeps the triage and downstream agents from ever assuming the absence of a symptom the patient was simply never asked about.

Together, these two changes turned the interview state from something regenerated and re-guessed every turn into something that accumulates reliably, turn after turn, the way an actual clinical intake would.

---

## Improving Reliability Through Structured Data

Another important design decision was to avoid relying exclusively on free-form model output. Free text is easy for a language model to generate, but it's brittle to validate, hard to store consistently, and difficult to reason over reliably in downstream logic like triage scoring or SOAP generation. Instead, every stage of the pipeline — intent classification, symptom extraction, triage assessment, and specialty recommendation — communicates through explicit, typed JSON schemas rather than unconstrained natural language.

For symptoms specifically, severity is never left as an open-ended description. It's constrained to a fixed set of values:

* `none`
* `mild`
* `moderate`
* `severe`

and pain is captured separately on a standardized 0–10 scale (`pain_scale_0_10`), alongside structured fields for location and type (sharp, dull, burning, unknown). Vital signs follow the same principle — each measurement (temperature, heart rate, blood pressure, respiratory rate, oxygen saturation) is either a concrete numeric value or one of a small set of controlled states (`unmeasurable`, `low`, `normal`, `high`, `unknown`), rather than a free-text field the backend would have to parse and interpret after the fact.

This constraint has several concrete benefits:

* **Validation becomes deterministic.** The backend can check that a field matches its expected type or enum rather than trying to infer meaning from arbitrary phrasing — a symptom is either `mild`, `moderate`, or `severe`, never something ambiguous like "kind of bad."
* **Merging and persistence stay reliable.** Because every field has a known shape, the delta-merging pipeline can update the interview state field-by-field with confidence, rather than guessing how a chunk of free text should be reconciled with what's already stored.
* **Downstream agents can consume the data directly.** The triage engine, the SOAP report generator, and the specialty recommender all operate on the same structured record, so a severity value or vital sign never needs to be re-parsed or reinterpreted differently by each stage.
* **The system stays predictable end-to-end.** Because every agent in the pipeline emits and consumes the same well-defined schema, the overall behavior of the system doesn't depend on the LLM's response format staying consistent from one call to the next.

In short, structuring the data wasn't just a data-modeling convenience — it's what made it possible for a chain of independent agents (classifier, extractor, triage engine, recommender, SOAP generator) to hand information to one another reliably, without each step silently degrading the quality of what came before it.

---

## 5. Triage and Urgency Assessment

Once enough information has been collected, Meddit evaluates the urgency of the situation.

The system categorizes cases into different urgency levels:

**None → Low → Medium → High → Crisis**

The resulting category influences the recommended next step.

For lower-risk situations, the system may provide appropriate general guidance.

For higher-risk situations, the system can recommend seeking professional medical care and identify an appropriate specialty.

For crisis-level situations, the system interrupts the conversational flow immediately and displays clear, actionable emergency guidance — instructing the patient to call an ambulance or contact someone nearby — rather than continuing with further interview questions. The platform does not place emergency calls on the patient's behalf; it is designed to prompt immediate human action rather than take that action itself.

The goal is not to replace a doctor or provide a definitive diagnosis.

Instead, the triage component is designed to help determine **what the patient should do next**.

![Triage System](/assets/img/meddit_triage.png)

---

## 6. Retrieval-Augmented Generation

A general-purpose language model does not automatically provide the controlled and consistent knowledge base that a medical application requires.

To address this, Meddit uses **Retrieval-Augmented Generation (RAG)**.

Relevant information is retrieved from a controlled knowledge base and provided to the language model as context when generating an answer.

The process can be summarized as:

**Patient question → Retrieval → Relevant medical information → LLM → Context-aware response**

This allows the generation component to ground its responses in the information available in the knowledge base rather than relying entirely on its pretrained knowledge.

It also provides a mechanism for controlling which sources of information are available to the system.

---

## 7. Context-Aware Responses

The final response generation stage considers multiple sources of information:

* The patient's current message.
* Previously collected medical history.
* Structured symptom information.
* The current interview state.
* Triage information.
* Retrieved medical knowledge.

This allows the system to generate responses that are specific to the ongoing interview rather than generic answers to isolated questions.

For example, the same question can require a different response depending on the symptoms and information already collected from the patient.

---

## 8. Doctor Handoff

Meddit is not designed to keep the patient inside an AI conversation indefinitely.

When professional medical care is recommended, the platform can connect the patient with an appropriate medical specialty and support communication with a doctor.

The system can also generate a **structured summary of the patient's interview**.

Instead of requiring the doctor to read the entire conversation, the summary organizes relevant information collected during the AI interview into a more useful format.

The AI-generated report covers the Subjective, Objective, and Assessment components — synthesized from the structured medical information, the triage output, the full conversation, and the recommended specialty — while the Plan is deliberately left for the doctor to complete, since treatment planning requires clinical judgment the AI is not positioned to provide. This draft is explicitly a starting point, not a final record: before a consultation can be marked complete, the doctor must review and edit the AI-generated draft — correcting, adding, or removing content as needed — and save it themselves. Only the doctor-edited version is persisted as the official record of the consultation. This keeps a clinician firmly in the loop on the platform's only artifact with lasting clinical and legal weight, consistent with Meddit's broader principle of AI as decision support rather than a replacement for professional judgment.


This creates a workflow in which AI assists with information gathering while a healthcare professional remains involved in the actual medical care.

![Doctor Dashboard](/assets/img/meddit_dashboard.png)

---

## System Architecture

Meddit was built as a distributed, multi-component platform rather than a single AI model wrapped in a chat interface. Each component has a distinct responsibility, communicating through well-defined APIs so that the AI pipeline, patient experience, and clinical workflow can evolve independently.

### Patient Application

The patient-facing experience is delivered through a **Flutter** mobile application, chosen for its ability to deliver a responsive, native-quality interface across platforms from a single codebase. Through the app, patients can:

* Start and continue a structured medical interview.
* Communicate naturally with the AI assistant.
* Review the information captured during their interview.
* Receive triage-based guidance on urgency and next steps.
* Book and join consultations with doctors.

### Backend

The system's backend is built on **FastAPI**, using asynchronous request handling throughout to keep the platform responsive under concurrent load — a requirement given that every patient interaction involves multiple sequential AI calls (intent classification, extraction, triage, and generation) rather than a single request-response cycle.

The backend is responsible for:

* Authentication and access control
* User and patient account management
* Medical interview orchestration and state
* Chat message handling
* Symptom and clinical data management
* Doctor profiles, availability, and scheduling
* Triage results and SOAP report generation
* Coordination of the AI agent pipeline
* All database operations

### Administrative & Clinical Interface

A web-based dashboard, built with **Angular**, gives doctors and administrators visibility into the platform. Doctors use it to review incoming SOAP summaries before a consultation, manage their schedules, and conduct online sessions, while administrators use it to monitor platform activity, manage accounts, and oversee system health.

### Data Layer

Persistent data is stored in a relational database **(PostgreSQL)** accessed through **SQLModel**, providing type-safe, schema-consistent access to structured records such as patients, interviews, symptoms, and doctors. Alongside this, a vector database supports the retrieval-augmented generation pipeline, enabling fast semantic search over the platform's medical knowledge base. Asynchronous data access is used throughout the API layer to keep read and write operations non-blocking, ensuring the system scales smoothly as concurrent interview sessions increase.

![meddit erd](</assets/img/meddit_erd (2).png>)

----
## Explainability

![Meddit](/assets/img/meddit_event_inspector_1.png)

![Meddit](/assets/img/meddit_event_inspector_2.png)

A recurring theme throughout Meddit's design was the rejection of the "black box" model of AI decision-making. In a clinical context, an urgency assessment or a specialty recommendation is only useful if the reasoning behind it can be inspected, questioned, and audited — by the patient, by the treating doctor, and by the team maintaining the system.

Rather than bolting explainability on as a logging afterthought, we treated it as an architectural constraint from the start, expressed through three mechanisms:

**Decision rationale, not just labels.** Both the triage agent and the specialty recommender are required to output structured reasoning alongside their conclusions. The triage agent doesn't simply return `high`; it returns a `chosen_logic` field describing why that severity was selected from the patient's symptoms, and an `exclusion_logic` field explaining why more severe or less severe classifications were ruled out. The specialty recommender behaves the same way, pairing its recommendation with a stated reason drawn from the patient's condition. This forces every categorical decision in the pipeline to carry its own justification, rather than leaving a doctor to reverse-engineer the model's reasoning from the output alone.

**Source traceability for retrieved information.** In the RAG path, the system never presents a generated answer without a link back to the underlying source. Every fact surfaced to the patient is tied to the specific document or passage it was retrieved from, which keeps general health answers verifiable and makes it straightforward to catch cases where a response has drifted from the retrieved context — a direct mitigation against hallucination in a domain where an ungrounded claim carries real risk.

**A complete, queryable audit trail.** Every stage of a patient's session — intent classification, information extraction, triage evaluation, clarification requests — is logged as a discrete, timestamped event with its full input and output payload. During development, this log became more than a debugging aid: we built an internal execution-path **inspector** that visualizes a session as a graph of agent calls, letting us step through exactly what a given agent received, what it produced, and how that fed the next stage. Beyond development, the same trail supports post-hoc clinical review and provides a foundation for detecting systematic bias or drift in the pipeline's decisions over time.

Together, these mechanisms mean that Meddit's outputs are traceable end-to-end: a doctor reviewing a SOAP summary can see not just the recommended specialty, but why it was recommended, and a developer debugging an unexpected triage result can reconstruct the exact chain of agent decisions that produced it.

---

## What I Learned

### Conversational AI Requires State

One of the biggest lessons from this project was that a medical chatbot cannot be treated as a simple sequence of independent prompts and responses. A clinically useful system needs an explicit model of what has already been established, what remains unknown, and how each new message should update that model — treating the conversation as a state machine rather than a stream of isolated exchanges. Once we adopted this framing, entire categories of bugs (repeated questions, contradictory follow-ups, lost context) simply stopped being possible by construction.

### Structured Outputs Matter

Language models are powerful at interpreting natural language, but application logic — validation, persistence, triage scoring, report generation — requires predictable, well-typed data. Enforcing structured schemas at every boundary between pipeline stages made the system dramatically easier to reason about, test, and integrate with the backend and database. In hindsight, the decision to constrain LLM outputs to explicit schemas early on was one of the highest-leverage design choices in the entire project.

### AI Engineering Is Different From Model Development

Building Meddit reinforced that working with LLMs in a production application is a systems engineering problem far more than a modeling problem. Selecting and prompting an LLM was a small fraction of the overall effort. The majority of the work went into the surrounding architecture:

* Multi-agent orchestration and routing logic
* API and database design
* Authentication and role-based access control
* Asynchronous request handling and WebSocket communication
* Input/output validation
* Persistent, mergeable conversation state
* Error handling and graceful degradation
* Monitoring, logging, and auditability

The AI model is one component embedded in a much larger application — and its reliability in practice depends almost entirely on the quality of that surrounding system.

### Medical AI Requires Conservative Design

Working on a healthcare-adjacent application sharpened our understanding of the boundary between **decision support** and **medical diagnosis**. Meddit is deliberately scoped to assist with information gathering, urgency assessment, and routing to appropriate care — not to diagnose or replace a clinician's judgment. This principle shaped concrete design decisions throughout the system: the strict `null`-versus-negative-finding distinction, the mandatory rationale fields on every triage decision, the confidence scoring on generated content, and the immediate, non-conversational escalation path for crisis-level cases. In a medical context, being conservative about what the AI is allowed to assert is as important as what it's technically capable of generating.

---

## Technologies

**AI & Machine Learning**

* Natural Language Processing (NLP)
* Large Language Models (LLMs)
* Intent Classification
* Structured Information Extraction
* Retrieval-Augmented Generation (RAG)
* Semantic Search / Dense Retrieval
* Generative AI
* Structured Outputs & Schema-Constrained Generation
* Agentic / Multi-Agent AI Architecture
* Explainable AI (XAI)

**Backend**

* Python
* FastAPI
* SQLModel
* PostgreSQL
* Vector Database
* WebSockets
* Asynchronous Programming
* JWT-based Authentication & RBAC

**Frontend**

* Flutter (patient mobile application)
* Angular (doctor & admin dashboard)

**Software Engineering**

* Clean / Decoupled Architecture
* SOLID Principles
* Design Patterns
* REST APIs
* Authentication & Authorization
* Input/Output Monitoring & Audit Logging

---

## What I Would Improve

There are several directions worth exploring in a future iteration of the project:

* Refine the medical interview policy and question-selection strategy, potentially using a more principled clinical decision-tree model rather than heuristic prompting.
* Expand and diversify the controlled medical knowledge base underpinning the RAG pipeline.
* Strengthen grounding and citation for retrieved medical information, including confidence-weighted source attribution.
* Build automated evaluation datasets and regression suites for intent classification and symptom extraction accuracy.
* Improve the structure and clinical usefulness of doctor-facing SOAP summaries based on direct physician feedback.
* Conduct formal evaluation of the system with healthcare professionals and realistic, adversarial patient scenarios.
* Explore tighter integration with emergency services or local emergency contacts for crisis-level cases, rather than relying solely on a static on-screen prompt.

---

## Conclusion

Meddit was an exploration of how modern AI techniques can be combined with disciplined software engineering to build a practical, production-oriented telehealth system. The project moved well beyond simply integrating an LLM into an application: it required designing a stateful medical interview, reliable structured information extraction, dynamic conversation flow, retrieval-augmented generation grounded in a controlled knowledge base, multi-level urgency assessment, an explainability layer capable of justifying every clinically relevant decision, and a complete backend and multi-platform application around all of these components.

The most important lesson from building Meddit was that useful AI systems are rarely defined by a single powerful model. They are defined by how models, structured data, business logic, and software architecture are made to work together reliably — turn after turn, decision after decision, in a domain where getting it wrong has real consequences.