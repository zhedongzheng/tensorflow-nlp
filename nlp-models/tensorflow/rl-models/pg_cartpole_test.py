from pg import PolicyGradient
import tensorflow as tf
import gym


def main():
    model = PolicyGradient(gym.make('CartPole-v1'),
                           n_in = 4,
                           hidden_net = lambda x : tf.layers.dense(x, 10, tf.nn.elu),
                           n_out = 2)
    model.learn()
    model.play()

if __name__ == '__main__':
    main()
