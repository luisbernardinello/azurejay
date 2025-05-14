<p align="center">
  <a href="#"><img src="assets/azurejay.png" height="250" /></a>
  <br/>
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

AzureJay, represented by an Atlantic Bird mascot, is a conversational language learning app whose mission is to provide a efficient conversational experience for language learning through AI-powered interactions using MRKL (Modular Reasoning, Knowledge, and Language) architecture. Our platform offers a refreshing alternative to traditional language learning apps by focusing on natural conversation.

Unlike conventional apps that rely solely on predefined lessons, AzureJay engages users in natural conversations while intelligently detecting errors in grammar and vocabulary, providing real-time corrections and explanations. The system combines neural network models with symbolic reasoning to deliver a more effective language learning experience.

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
      <a href="#mrkl-architecture">MRKL Architecture</a>
      <ul>
        <li><a href="#architecture-overview">Architecture Overview</a></li>
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
- 🔍 **Intelligent Error Detection**: Real-time grammar and vocabulary
- 🧠 **MRKL Architecture**: Hybrid neuro-symbolic system combining neural networks with symbolic reasoning for more accurate explanations
- 🎯 **Personalized Learning**: Adapts to your interests
- 📊 **Progress Tracking**: Monitor your improvement over time with detailed analytics
- 🌐 **Multi-Language Support**: Initially supporting English, with more languages planned

## MRKL Architecture

### Architecture Overview

AzureJay is built on the MRKL (Modular Reasoning, Knowledge, and Language) architecture, a cutting-edge neuro-symbolic approach that combines the strengths of neural networks with symbolic reasoning systems:

```
                                                      +-----------+
                                                      | __start__ |
                                                      +-----------+
                                                             *
                                                             *
                                                             *
                                                  +-------------------+
                                               ...| ai_language_tutor |....
                                     ..........  *+-------------------+.   .........
                          ...........       *****            *          .....       ..........
                ..........             *****                *                ....             .........
          ......                    ***                     *                    ...                   .....
+---------+           +----------------+           +----------------+           +--------------+           +----------------+
| __end__ |           | update_grammar |           | update_profile |           | update_topic |           | web_search_api |
+---------+           +----------------+           +----------------+           +--------------+           +----------------+

```

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
- Run `docker compose up --build`
- Run `docker compose down` to stop all services

#### How to run tests.

- Run `pytest` to run all tests

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
- [ ] 🏁 KMP application development
- [ ] 🏁 Enhanced pronunciation assessment
- [ ] 🏁 Cultural context integration in conversations
- [ ] 🏁 Spaced repetition based on conversation history

## Roadmap

- [ ] Implement dialect recognition and support
- [ ] Voice-assisted hands-free learning experience
- [ ] Community features for peer language exchange
- [ ] AR/VR immersive learning experiences

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
