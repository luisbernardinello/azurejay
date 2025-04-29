<p align="center">
  <a href="#"><img src="img/azurejay.png" height="250" /></a>
  <br/><br/>
  <font size="6"><b>AzureJay</b></font>
  <br/>
  <em>A neuro-symbolic conversation-based language learning platform</em>
  <br/><br/>
  <a href="#"><img src="https://img.shields.io/badge/Try_now-azurejay.app-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/List_of_courses-azurejay.app/courses-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Documentation-azurejay.app/docs-darkcyan" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Roadmap-github.com-darkcyan" /></a>
</p>

<hr/>

## About

AzureJay, represented by an Atlantic Bird mascot, is a conversational language learning app whose mission is to provide a efficient conversational experience for language learning through AI-powered interactions using MRKL (Modular Reasoning, Knowledge, and Language) neuro-symbolic architecture. Our platform offers a refreshing alternative to traditional language learning apps by focusing on natural conversation, precise grammar correction, contextual feedback, and adaptive learning paths tailored to each user's proficiency level and interests.

Unlike conventional apps that rely solely on predefined lessons, AzureJay engages users in natural conversations while intelligently detecting errors in grammar, pronunciation, and vocabulary, providing real-time corrections and explanations. The system combines neural network models with symbolic reasoning to deliver a more effective language learning experience.

All software is licensed under AGPLv3, which guarantees the freedom to run, study, share, and modify the software.

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#features">Key Features</a>
    </li>
    <li>
      <a href="#mrkl-architecture">MRKL Architecture</a>
      <ul>
        <li><a href="#architecture-overview">Architecture Overview</a></li>
        <li><a href="#core-modules">Core Modules</a></li>
        <li><a href="#how-it-works">How It Works</a></li>
      </ul>
    </li>
    <li>
      <a href="#using-azurejay">Using AzureJay</a>
      <ul>
        <li><a href="#web-app">Web App</a></li>
        <li><a href="#mobile-app">Mobile App (Coming Soon)</a></li>
        <li><a href="#azurejay-api">AzureJay API</a></li>
      </ul>
    </li>
    <li>
      <a href="#developing-azurejay">Developing AzureJay</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#getting-started">Getting Started</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
      </ul>
    </li>
    <li>
     <a href="#milestones">Milestones</a>
    </li>
    <li>
     <a href="#roadmap">Roadmap</a>
    </li>
    <li>
     <a href="#contribution">Contribution</a>
     <ul>
        <li><a href="#become-a-contributor">Become a Contributor</a></li>
        <li><a href="#attributions">Attributions</a></li>
        <li><a href="#contributors">Contributors</a></li>
     </ul>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#see-also">See Also</a>
    </li>
    <li>
      <a href="#donate">Donate</a>
    </li>
  </ol>
</details>

## Features

AzureJay offers a unique language learning experience with these key features:

- 🗣️ **Conversation-Based Learning**: Natural dialogue with AI language partners adjusted to your proficiency level
- 🔍 **Intelligent Error Detection**: Real-time grammar, vocabulary, and pronunciation correction
- 📝 **Contextual Explanations**: Clear, educational explanations for corrections tailored to your level
- 🧠 **MRKL Architecture**: Hybrid neuro-symbolic system combining neural networks with symbolic reasoning for more accurate language understanding
- 🎯 **Personalized Learning**: Adapts to your interests, learning patterns, and proficiency level
- 📊 **Progress Tracking**: Monitor your improvement over time with detailed analytics
- 🔄 **Multi-Device Synchronization**: Seamlessly continue learning across different devices
- 📱 **Cross-Platform**: Works on various devices including phones, tablets, and desktop computers
- 👥 **Community-Driven**: Open-source and community-owned development
- 🌐 **Multi-Language Support**: Initially supporting English and Portuguese, with more languages planned

## MRKL Architecture

### Architecture Overview

AzureJay is built on the MRKL (Modular Reasoning, Knowledge, and Language) architecture, a cutting-edge neuro-symbolic approach that combines the strengths of neural networks with symbolic reasoning systems:

```
┌───────────────────┐
│  User Input       │
│  (Text/Voice)     │
└─────────┬─────────┘
          │
┌─────────▼──────────┐
│ Intelligent Router │
└─────────┬──────────┘
          │
┌─────────▼─────────┐      ┌─────────────────────┐      ┌───────────────────┐
│ MRKL Modules      │◄────►│ Knowledge Retrieval │◄────►│ Language Models   │
└─────────┬─────────┘      └─────────────────────┘      └───────────────────┘
          │
┌─────────▼─────────┐
│ Response          │
│ Orchestrator      │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│ User Response     │
└───────────────────┘
```

### Core Modules

AzureJay's MRKL system consists of several specialized modules:

1. **Input Processing Module**

   - Handles voice transcription via Whisper ASR
   - Normalizes and preprocesses text input

2. **Intelligent Router**

   - Uses RoBERTa-based intent classification
   - Analyzes user intent to route to appropriate expert modules
   - Supports multi-module activation when needed

3. **Neuro-Symbolic Grammar Correction Module**

   - Combines T5-based grammar correction with symbolic rule-based explanations
   - Provides educational explanations for corrections

4. **Conversation Generation Module**

   - Uses LLMs (currently GPT-4) with structured prompting
   - Generates responses appropriate to the user's proficiency level

5. **Knowledge Retrieval Module**

   - Integrates Elasticsearch and Wikidata API
   - Provides factual information for language-related questions
   - Fall backs to LLM for complex queries

6. **User Personalization Module**

   - Tracks user interests, proficiency levels, and conversation patterns
   - Detects topic exhaustion and suggests new conversation themes
   - Adapts content difficulty based on user progress

7. **Response Orchestrator**
   - Combines outputs from different expert modules
   - Prioritizes corrections and explanations when needed
   - Ensures cohesive and educational final responses

### How It Works

1. **User Interaction**: Users speak or type in the target language
2. **Intelligent Routing**: The system analyzes the input to determine if it needs correction, factual knowledge, or conversational response
3. **Multi-Module Processing**: Relevant expert modules process the input simultaneously
4. **Orchestrated Response**: The system combines the outputs from different modules, prioritizing educational content
5. **Adaptive Learning**: The system tracks user progress to personalize future interactions

## Using AzureJay

### Web App

AzureJay offers an official [web app](https://azurejay.app/), which leverages the MRKL architecture for neuro-symbolic natural language processing. Try out AzureJay language learning through conversation right in your browser, no installation required!

#### Screenshots

<p align="center">
  <img src="/docs/screenshots/screenshot1.png" width="15%" />
  <img src="/docs/screenshots/screenshot2.png" width="15%" />
  <img src="/docs/screenshots/screenshot3.png" width="15%" />
  <img src="/docs/screenshots/screenshot4.png" width="15%" />
</p>

### Mobile App (Coming Soon)

We're developing a Flutter-based mobile application that will provide:

- Voice interaction for pronunciation practice
- Personalized learning paths and progress tracking
- Cross-platform support for iOS and Android

### AzureJay API

For developers interested in integrating AzureJay's capabilities into their own applications, we offer a comprehensive API:

```python
import azurejay

# Initialize the MRKL system
mrkl_system = azurejay.init_mrkl_pipeline(config_path="./config.json")

# Process user message
response = mrkl_system.process_message(
    user_id="user123",
    message="I goed to the store yesterday",
    voice_mode=False
)

print(response)
# Output: I notice you said: 'I goed to the store yesterday'.
# A better way to say this would be: 'I went to the store yesterday'.
# For past tense of 'go', we use the irregular form 'went' and not 'goed'.
```

## Developing AzureJay

### Prerequisites

- Python 3.8+
- Required Python packages (see requirements.txt)
- Flutter SDK (for frontend development)
- ElasticSearch (for knowledge retrieval system)

### Getting Started

1. Clone the repository

```bash
git clone https://github.com/luisbernardinello/AzureJay.git
cd AzureJay
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the MRKL pipeline tests

```bash
python tests/test_mrkl_pipeline.py
```

For Flutter development (coming soon):

```bash
cd flutter_app
flutter pub get
flutter run
```

### Project Structure

```
AzureJay/
├── mrkl/
│   ├── __init__.py
│   ├── router.py                # Intelligent routing module
│   ├── correction.py            # Grammar correction module
│   ├── knowledge.py             # Knowledge retrieval module
│   ├── personalization.py       # User modeling module
│   ├── generation.py            # Response generation module
│   └── orchestrator.py          # Response orchestration logic
├── models/
│   ├── __init__.py
│   ├── intent_classifier/       # RoBERTa intent classifier
│   ├── grammar_corrector/       # T5-based grammar correction
│   └── user_model.py            # User profiling and tracking
├── utils/
│   ├── __init__.py
│   ├── audio.py                 # Audio processing utilities
│   └── text.py                  # Text normalization utilities
├── api/
│   ├── __init__.py
│   ├── routes.py                # API endpoints
│   └── server.py                # API server implementation
├── flutter_app/                 # Flutter mobile application (coming soon)
├── tests/
│   ├── test_mrkl_pipeline.py
│   ├── test_router.py
│   └── test_correction.py
├── requirements.txt
├── config.json                  # Configuration settings
└── README.md
```

## Milestones

- [x] 🏁 Core MRKL architecture implementation
- [x] 🏁 Intelligent router for module selection
- [x] 🏁 Neuro-symbolic grammar correction system
- [x] 🏁 Basic conversation generation
- [x] 🏁 Knowledge retrieval integration
- [x] 🏁 User profiling and personalization
- [x] 🏁 Response orchestration system
- [x] 🏁 Basic theme/branding
- [x] 🏁 Initial language support for English and Portuguese
- [ ] 🏁 Flutter mobile application development
- [ ] 🏁 Enhanced pronunciation assessment
- [ ] 🏁 Cultural context integration in conversations
- [ ] 🏁 Spaced repetition based on conversation history

## Roadmap

- [ ] Add support for Asian languages (Japanese, Mandarin, Korean)
- [ ] Implement dialect recognition and support
- [ ] Voice-assisted hands-free learning experience
- [ ] Offline mode for mobile applications
- [ ] Community features for peer language exchange
- [ ] AR/VR immersive learning experiences
- [ ] Specialized vocabulary modules for different professions

### Projects

- [Improved pronunciation feedback system](https://github.com/luisbernardinello/AzureJay/projects/1)
- [Cultural context integration](https://github.com/luisbernardinello/AzureJay/projects/2)
- [Flutter application development](https://github.com/luisbernardinello/AzureJay/projects/3)

## Contribution

### Become a Contributor

#### Are you a developer?

You can help AzureJay by testing it and submitting feature requests or bug reports: [here](https://github.com/luisbernardinello/AzureJay/issues/new). If you want to get in touch, you can use my contact details on [my GitHub profile](https://github.com/luisbernardinello).
Go through the dev docs [here](https://azurejay.app/docs/CONTRIBUTING.html).

Our current development priorities include:

- Flutter app implementation
- Improving grammar correction accuracy
- Expanding language support
- Enhancing pronunciation feedback

Still got questions? Our Matrix channel is `#AzureJay`, join the dev community there and feel free to ask anything.

- Matrix: [`#AzureJay`](https://app.element.io/#/room/#space-azurejay:matrix.org) on `matrix.org`

#### Are you a linguist or language teacher?

Help us improve our language models and conversational patterns! We particularly need help with:

- Creating grammar rule explanations
- Developing conversational prompts appropriate for different proficiency levels
- Validating our error detection system
- Creating cultural context for language learning

We are continuously working to improve the learning experience. If you have ideas to make it better, please share them with us by creating an [issue](https://github.com/luisbernardinello/AzureJay/issues/new).

### Attributions

#### Mascot

The mascot is designed by [@luisbernardinello](https://github.com/luisbernardinello). If you have ideas to make it better, please share them with us by creating an [issue](https://github.com/luisbernardinello/AzureJay/issues/new).

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Mascot images are released under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

### Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->

[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors-)

<!-- ALL-CONTRIBUTORS-BADGE:END -->

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/luisbernardinello"><img src="https://avatars.githubusercontent.com/u/162613265?v=4" width="100px;" alt="Luis Bernardinello"/><br /><sub><b>Luis Bernardinello</b></sub></a><br /><a href="https://github.com/luisbernardinello/AzureJay/commits?author=luisbernardinello" title="Code">💻</a></td>
      <!-- Add other contributors as needed -->
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

AzureJay is licensed under the AGPL-3.0 license. In addition, course content and other creative content might be licensed under different licenses, such as CC.

## See Also

- [Duolingo](https://www.duolingo.com/), gamified language learning
- [Tandem](https://www.tandem.net/), language exchange with real people
- [italki](https://www.italki.com/), connect with language teachers

## Donate

Help us to keep going and expand our language offerings by supporting the project through [our donation page](https://azurejay.app/donate).

Your donations help us:

- Maintain server infrastructure
- Develop new language modules
- Improve our AI models
- Keep AzureJay free and open source
