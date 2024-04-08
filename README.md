# AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via Consensus-based Multi-Agent Reinforcement Learning

![Experimental Results Image](https://github.com/mao1207/Steering-Stakeholder-Dynamics-in-Urban-Planning-via-Consensus-based-MARL/blob/main/imgs/Actor_Critic_framework_1.jpg?raw=true)


## Abstract
In urban planning, land use readjustment plays a pivotal role in aligning land use configurations with the current demands for sustainable urban development. However, present-day urban planning practices face two main issues. Firstly, land use decisions are predominantly dependent on human experts. Besides, while resident engagement in urban planning can promote urban sustainability and livability, it is challenging to reconcile the diverse interests of stakeholders. To address these challenges, we introduce a Consensus-based Multi-Agent Reinforcement Learning framework for real-world land use readjustment. This framework serves participatory urban planning, allowing diverse intelligent agents as stakeholder representatives to vote for preferred land use types. Within this framework, we propose a novel consensus mechanism in reward design to optimize land utilization through collective decision making. To abstract the structure of the complex urban system, the geographic information of cities is transformed into a spatial graph structure and then processed by graph neural networks. Comprehensive experiments on both traditional top-down planning and participatory planning methods from real-world communities indicate that our computational framework enhances global benefits and accommodates diverse interests, leading to improved satisfaction across different demographic groups. By integrating Multi-Agent Reinforcement Learning, our framework ensures that participatory urban planning decisions are more dynamic and adaptive to evolving community needs and provides a robust platform for automating complex real-world urban planning processes.

## Environment Setup

To run BG-HGNN, ensure that you have the following environments:

- PyTorch 2.0.0+
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- DGL

We recommend installing the DGL library, which is required for BG-HGNN. For installation commands, please visit [DGL Start Page](https://www.dgl.ai/pages/start.html).

## Kendall Square Test
The Kendall Square dataset is a city dataset we organized based on the current urban conditions of Kendall Square. It includes 748 blocks and 60 voting nodes. We have set up five methods, including random voting, centralized greedy, decentralized greedy, centralized MARL, and decentralized MARL. You can run the models with the following command:

```shell
python model/main.py --experiment_mode your_model --if_only_top_down decentralized_or_centralized
```

![Experimental Results Image](https://github.com/mao1207/Steering-Stakeholder-Dynamics-in-Urban-Planning-via-Consensus-based-MARL/blob/main/imgs/Kendall_Square_result.png?raw=true)

## Urbanity Global Network Test
You can download the dataset from [Urbanity](https://github.com/winstonyym/urbanity), which includes data from 50 cities from different regions. You can run `Urbanity/get_city_graphs.py` for processing, including getting 50 global city graphs and their subgraphs. You can choose to train on these subgraphs or the global city graph and eventually test the results on the complete graph.

![Experimental Results Image 2](https://github.com/mao1207/Steering-Stakeholder-Dynamics-in-Urban-Planning-via-Consensus-based-MARL/blob/main/imgs/Urbanity.png?raw=true)
"""
