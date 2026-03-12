# Keboli – Interview Agent

The **Interview Agent** is the AI-driven service responsible for conducting interview sessions, interacting with candidates, generating interview questions, and building a **skill graph** using Large Language Models (LLMs).

This service works alongside the backend API and real-time communication infrastructure to simulate a structured technical interview.


---

# 1. Project Information

## Overview

The Interview Agent acts as the **AI interviewer** in the AI Interview Bot platform.

It connects to a real-time communication session and interacts with the candidate by asking questions and collecting responses.

The agent integrates with **LLM services** to dynamically generate questions and build a **skill graph** that guides the interview flow.

Based on the candidate's responses, the agent determines the next question to ask.

The collected responses are later processed by the **Evaluation Agent**, which performs detailed candidate assessment.

---

## Responsibilities

The Interview Agent is responsible for:

* conducting AI-driven interviews
* generating interview questions
* generating a **skill graph** for interview flow
* dynamically selecting the next question
* collecting candidate responses
* interacting with the real-time communication system
* sending collected responses to the backend

---

## Key Features

* AI-based interview orchestration
* dynamic question generation using LLMs
* skill graph generation for structured interviews
* real-time interaction with candidates
* adaptive questioning based on candidate responses
* integration with LiveKit for session communication
* asynchronous task handling

---

## Technology Stack

| Component            | Technology |
| -------------------- | ---------- |
| Language             | Python     |
| AI Orchestration     | LangGraph  |
| LLM Provider         | Groq       |
| Realtime Integration | LiveKit    |
| HTTP Client          | httpx      |
| Async Framework      | asyncio    |

---

# 2. Architecture Overview

The Interview Agent operates as a **separate microservice** that communicates with the backend and AI model providers.


## System Architecture

```
                   +------------------------+
                   |       FastAPI Backend  |
                   |  (Session Management)  |
                   +-----------+------------+
                               |
                               | Start Interview
                               |
                               v
                    +-----------------------+
                    |    Interview Agent    |
                    |                       |
                    |  - Interview Logic    |
                    |  - Question Generation|
                    |  - Skill Graph Build  |
                    +-----------+-----------+
                                |
                ----------------------------------
                |                                |
                v                                v

        +----------------+              +------------------+
        |   LiveKit      |              |   LLM Provider   |
        | Real-time Room |              |      (Groq)      |
        +----------------+              +------------------+

```

---

## Component Responsibilities

### Interview Agent Core

Handles:

* interview workflow orchestration
* dynamic question sequencing
* skill graph generation
* response collection

---

### LiveKit Integration

The agent connects to **LiveKit rooms** to interact with candidates.

Used for:

* audio/video communication
* real-time session events

---

### LLM Adapter

The LLM adapter module:

* generates interview questions
* builds the skill graph
* determines the next question based on responses
* manages adaptive interview flow

---

### Backend Communication

The agent communicates with the backend to:

* receive interview session information
* send collected candidate responses
* update interview session status

---

# 3. Service Interaction Flow

```
Recruiter Creates Assessment
            |
            v
Backend API
            |
            | Candidate joins interview
            v
LiveKit Session Created
            |
            v
Interview Agent Connects
            |
            | Generate Skill Graph
            |
            v
Agent Generates Questions
            |
            v
Candidate Responds
            |
            v
Agent Collects Responses
            |
            v
Responses Stored in Backend
            |
            v
Evaluation Agent Processes Responses
            |
            v
Evaluation Results Generated
```

---


# 4. Interview Workflow

The Interview Agent follows a structured workflow:

### 1. Session Initialization

* backend triggers interview session
* agent connects to the LiveKit room

---

### 2. Skill Graph Generation

The agent generates a **skill graph** based on:

* assessment configuration
* required technical skills
* interview structure

This graph determines how the interview will progress.

---

### 3. Question Generation

The agent generates interview questions using LLMs based on:

* skill graph structure
* candidate responses
* interview difficulty progression

---

### 4. Candidate Interaction

The candidate answers questions during the session.

Responses are collected and stored.

---

### 5. Response Collection

The agent collects and sends candidate responses to the backend service.

---



# 5. Running the Interview Agent Locally

## Prerequisites

Install:

* Python 3.10+
* Docker (optional)
* LiveKit server or LiveKit Cloud account

---

# Option 1 – Run with Docker

Clone repository:

```
git clone https://github.com/JeneshaMalar/keboli-interview-agent.git
cd keboli-interview-agent
```

Build and run container:

```
docker build -t interview-agent .
docker run interview-agent
```

---

# Option 2 – Run without Docker

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Linux / Mac

```
source venv/bin/activate
```

Install dependencies:

```
uv sync
```

---

### Start Interview Agent

Entry point:

```
fastapi_server.py
```

Run service:

```
bash start.sh
```

---

# 6. Environment Variables

Example `.env` file:

```
MAIN_BACKEND_URL=http://localhost:8000/api
DEEPGRAM_API_KEY=<YOUR API KEY>
GROQ_API_KEY=<YOUR API KEY>
GROQ_MODEL=llama-3.3-70b-versatile
LIVEKIT_URL=wss://<YOUR PROJECT>.livekit.cloud
LIVEKIT_API_KEY=<YOUR API KEY>
LIVEKIT_API_SECRET=<YOUR SECRET KEY>
BEY_API_KEY=<YOUR API KEY>
```

---

# 7. Error Handling and Logging

The Interview Agent includes logging mechanisms for:

* interview session lifecycle
* LLM communication errors
* API failures

System logs help monitor interview execution and diagnose issues.

---

# Author

**Jenesha Malar**

Keboli – Interview Agent Development

