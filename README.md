<p align="center">
  <a href="#"><img src="assets/azurejay.png" height="250" /></a>
  <br/>
  <font size="6"><b>AzureJay</b></font>
  <br/>
  <em>A conversation-based language learning app</em>
  <br/><br/>
  <a href="#"><img src="https://img.shields.io/badge/Try_now-azurejay.app-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/List_of_courses-azurejay.app/courses-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Documentation-azurejay.app/docs-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Roadmap-github.com-darkcyan" /></a>
</p>

<hr/>

## About

AzureJay, represented by an Atlantic Bird mascot, is a conversational language learning app whose mission is to provide an efficient conversational experience for language learning through AI-powered interactions using an intelligent Supervisor architecture.

Unlike conventional apps that rely solely on predefined lessons, AzureJay engages users in natural conversations while orchestrating multiple specialized agents to detect errors in grammar and vocabulary, gather contextual information, and provide corrections and explanations. The system combines neural network models with symbolic reasoning through a multi-agent workflow to deliver a conversational language learning experience.

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#features">Features</a>
    </li>
    <li>
      <a href="#supervisor-architecture">Supervisor Architecture</a>
      <ul>
        <li><a href="#architecture-overview">Architecture Overview</a></li>
        <li><a href="#specialized-agents">Specialized Agents</a></li>
        <li><a href="#workflow-orchestration">Workflow Orchestration</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a></li>
    <li>
     <a href="#milestones">Milestones</a>
    </li>
    <li>
     <a href="#roadmap">Roadmap</a>
    </li>
    <li>
     <a href="#contribution">Contribution</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#donate">Donate</a>
    </li>
  </ol>
</details>

## Features

AzureJay offers a Voice-Powered App with these key features:

- 🗣️ **Conversation-Based Learning**: Natural dialogue with AI tutor
- 🔍 **Multi-Layer Error Detection**: Real-time grammar and semantic correction using LanguageTool API and LLM verification
- 🧠 **Supervisor Architecture**: Intelligent workflow orchestration with specialized agents for different tasks
- 🎯 **Personalized Learning**: Adapts to your interests and maintains long-term memory of your progress
- 📊 **Progress Tracking**: Monitor your improvement over time with detailed analytics
- 🔗 **Contextual Research**: Web search integration for real-time information gathering
- 💾 **Persistent Memory**: Long-term storage of user profiles and grammar correction history

## Supervisor Architecture

### Architecture Overview

AzureJay is built on an intelligent Supervisor architecture that orchestrates multiple specialized agents to provide comprehensive language learning support:

```
                                                      +-----------+
                                                      | __start__ |
                                                      +-----------+
                                                             *
                                                             *
                                                             *
                                                  +-------------------+
                                               ...| supervisor        |....
                                     ..........  *+-------------------+.   .........
                          ...........       *****            *          .....       ..........
                ..........             *****                *                ....             .........
          ......                    ***                     *                    ...                   .....
+---------+           +----------------+           +----------------+           +--------------+           +----------------+
| __end__ |           | correction     |           | researcher     |           | responder    |           | web_search_api |
+---------+           +----------------+           +----------------+           +--------------+           +----------------+
```

### Specialized Agents

The Supervisor coordinates the following specialized agents:

#### 🎯 **Supervisor Agent**

- **Role**: Central workflow orchestrator
- **Responsibilities**:
  - Routes tasks to appropriate specialists based on current state
  - Ensures efficient workflow without unnecessary steps
  - Maintains transparency in decision-making process

#### ✏️ **Correction Agent**

- **Role**: Multi-layer grammar and semantic correction specialist
- **Layer 1 - Syntax Analysis**: Uses LanguageTool API for comprehensive grammar checking
- **Layer 2 - Semantic Verification**: Employs LLM verification for contextual and semantic corrections
- **Layer 3 - Synthesis**: Combines suggestions from both layers for optimal corrections

#### 🔍 **Research Agent**

- **Role**: Information gathering and fact-finding specialist
- **Capabilities**:
  - Web search integration using Tavily Search API
  - Updated information retrieval
  - Source credibility assessment
  - Structured information presentation
- **Activation**: Called only when users ask direct questions

#### 🤖 **Responder Agent (Subgraph)**

- **Role**: Conversational AI tutor with memory management
- **Components**:
  - **Memory Management**: Long-term storage of user profiles and correction history
  - **Personalized Responses**: Adapts to user interests and learning progress
  - **TrustCall Integration**: Structured information extraction and storage
  - **Contextual Awareness**: Incorporates corrections and research into responses

### Workflow Orchestration

1. **Input Processing**: User message enters the system
2. **Grammar Correction**: Supervisor routes to Correction Agent
3. **Research Evaluation**: Determines if external information gathering is needed
4. **Response Generation**: Routes to Responder subgraph for final interaction
5. **Memory Updates**: Automatically saves user profiles and correction history

**Key Benefits:**

- **Accuracy**: Multi-layer correction ensures high-quality feedback
- **Personalization**: Long-term memory enables personalized learning experiences

### Getting Started

#### Clone the repository

```bash
git clone https://github.com/luisbernardinello/AzureJay.git
cd AzureJay
```

#### Install all dependencies.

- Run `pip install -r requirements-dev.txt`

#### How to run app. Using Docker with PostgreSQL.

- Install Docker Desktop
- Run `docker-compose up --build`
- Run `docker-compose down` to stop all services

#### How to run tests.

- Run `pytest` to run all tests

## Milestones

- [x] 🏁 Supervisor architecture implementation
- [x] 🏁 Multi-layer grammar correction with LanguageTool integration
- [x] 🏁 Semantic verification using LLM models
- [x] 🏁 Research agent with web search capabilities
- [x] 🏁 Memory management with TrustCall integration
- [x] 🏁 User profiling and personalization system
- [x] 🏁 Initial language support for English
- [ ] 🏁 KMP application development
- [ ] 🏁 Enhanced pronunciation assessment

## Roadmap

- [ ] Cultural context integration in conversations
- [ ] Spaced repetition based on conversation history
- [ ] Additional specialized agents for pronunciation and fluency
- [ ] Multi-language supervisor architecture support

## Contribution

### Become a Contributor

#### Are you a developer?

You can help AzureJay by testing it and submitting feature requests or bug reports: [here](https://github.com/luisbernardinello/AzureJay/issues/new). If you want to get in touch, you can use my contact details on [my GitHub profile](https://github.com/luisbernardinello).

We are continuously working to improve the learning experience. If you have ideas to make it better, please share them with us by creating an [issue](https://github.com/luisbernardinello/AzureJay/issues/new).

### Attributions

#### Mascot

The mascot is designed by me [@luisbernardinello](https://github.com/luisbernardinello). If you have ideas to make it better, please share them with us by creating an [issue](https://github.com/luisbernardinello/AzureJay/issues/new).

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Mascot images are released under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/luisbernardinello"><img src="https://avatars.githubusercontent.com/u/162613265?v=4" width="100px;" alt="Luis Bernardinello"/><br /><sub><b>Luis Bernardinello</b></sub></a><br /><a href="https://github.com/luisbernardinello/AzureJay/commits?author=luisbernardinello" title="Code">💻</a></td>
      <!--  other contributors-->
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

AzureJay is licensed under the AGPL-3.0 license. In addition, course content and other creative content might be licensed under different licenses, such as CC.

## Donate

Help us to keep going and expand our language offerings by supporting the project through [our donation page](https://azurejay.app/donate).
