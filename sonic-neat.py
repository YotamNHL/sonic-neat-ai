import gym
import numpy as np
import cv2
import neat
import pickle
from time import sleep
from PIL import Image
import cv2
import retro
from visualize import *
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        obs = env.reset()
        x, y, color = env.observation_space.shape

        x = int(x / 6)
        y = int(y / 6)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        current_fitness = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        obs2 = 0
        while not done:
            # sleep(0.008)
            # env.render()
            frame += 1
            obs2 = cv2.resize(obs, (y, x))
            obs2 = cv2.cvtColor(obs2, cv2.COLOR_BGR2GRAY)
            cv2.imshow('i', obs2)

            obs = cv2.resize(obs, (x, y))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (x, y))

            imgarray = obs.flatten()
            nnOutput = net.activate(imgarray)

            obs, reward, done, info = env.step(nnOutput)
            xpos = info['x']


            if xpos > xpos_max:
                current_fitness += 1
                xpos_max = xpos
            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                counter = 0
            else:
                counter += 1
            if xpos == 2097:
                current_fitness += 50000
                xpos_max == xpos

            if counter == 300:
                done = True
                print(genome_id, current_max_fitness)

            genome.fitness = current_fitness



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, 'config-feedforward')
p = neat.Population(config)

p2 = neat.checkpoint.Checkpointer.restore_checkpoint('./neat-checkpoint-626')
p2.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p2.add_reporter(stats)
p2.add_reporter(neat.Checkpointer(50))
winner = p2.run(eval_genomes)
# winner = p.run(eval_genomes)

draw_net(config, winner, True)
plot_stats(stats, ylog=False, view=True)
plot_species(stats, view=True)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
