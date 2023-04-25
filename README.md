# LocalAttentionNCA
A neural cellular automata controlled by a network incorporating a attention layer that can use local information.

## Motivation and Background

Neural cellular automata can learn various behaviors, like controlling a cart pole (https://avariengien.github.io/self-organized-control/) and self-organize into thumbnails (https://distill.pub/2020/growing-ca/).

Neural cellular automata, as part of the artificial life research area, are used to investigate open-ended behavior. The CA in this research are often continous, but we use a discrete CA. Part of such setups is often a multidimensional CA: Additionally to an "alpha" channel, which models the aliveness of cells, similar to the classical CA. Additionally, this setup has an energy and a chemistry layer.

Part of open-ended behavior is the question, how much of a general purpose AI can be constructed. This approach is followed in this work, testing out some ideas. The idea is to train an AI that has some ability to seek paths through the CA, seeking out energy sources.

## Methods

Can an agent in a cellular automata which is controlled by a single neural network with local information be trained to seek out energy sources and thus navigate around the grid and thus survive for a prolonged time?

### Description of the Neural Cellular Automata

The cellular automata, as mentioned, has an alpha channel, an energy channel, a hidden channel (currently unused), and the alpha channel. The neural network consists of three layers, similar to the architecture in Growing Neural Cellular Automata. It operates locally: The channels of a single cell, and its eight neighboring cells, are the input to the neural network. The network computes an update for the alpha channel, which is the applied to the CA. Only the cells which are "alive", e.g. with an alpha channel over 0.1 and its neighbors are updated. The alpha channels are clipped to values between 0.1 and 1.
The energy channel is initialized by the user and can be changed during the training. The values of the energy are updated deterministically, depending on the alpha channel: Energy, which is in on a cell with an alive cell, is distributed in the neighboring cell according to the alpha channels. Similarly the chemistry channel is updated: On every alive cell, the chemistry value is increased by 0.1, on cells with no alive cell the values is decreased by 0.1 up to a value of 0. Those deterministic updates happen after the update of the alpha channel of the neural network.
The first layer is a self-attention layer. As input it takes the nine cells, with all channels, around/including the current cell. This means, that the input to the neural network is a $9 \times 4$ array (mind that the hidden channel is currently unused and thus 0). Next comes two linear layers, the last layers output is a $1 \times 4$ array. This procedure loops through all alive cells.

### Implementation Details

### Visualization
