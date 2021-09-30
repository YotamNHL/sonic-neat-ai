## What's that
This is a demonstration of Reinforcement learning. I've used NEAT (NeuroEvolution of Augmenting Topologies) on top of OpenAI's retro framework, in order to create an artificial neural network that will teach itself over time how to play the first level of the original Sonic The Hedgehog.

## Show me!
https://user-images.githubusercontent.com/46316863/135294270-18814f4a-7584-462a-9ae8-f6b992b94d94.mp4

## How exactly does it work?
On the left of the screen you can view the stream input the agent injects into its network (greyscaled and downscaled to make it run faster), and for each frame/input the network decides on an output - which is basically a sequence of button mashing in the game. At the initiation, The process creates "agents" with random outputs, and calculates fitness according to a predetermined "reward" system. In this specific usecase, I've programmed the agent to favour increasing is x value (how far to the right Sonic gets) and to avoid wasting time. After the initial state, this algorithm creates new genomes over the multiple generations of the process, and creates new genomes that are based  on the previews ones that did better, and also create ones that correlate with one-another, but with some random mutations over time. Eventually this process will create genomes that are able to get better and better results. What you see here is the result of 600 generations.
