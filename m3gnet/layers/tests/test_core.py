import unittest

import numpy as np
import tensorflow as tf
from pymatgen.core.structure import Lattice, Structure

from m3gnet.graph import RadiusCutoffGraphConverter
from m3gnet.layers import MLP, Embedding, GatedMLP, Pipe


class _Layer(tf.keras.layers.Layer):
    def call(self, x, **kwargs):
        return x * x


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gc = RadiusCutoffGraphConverter(5)
        s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s2 = Structure(Lattice.cubic(3), ["Mo", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.g1 = gc.convert(s1)
        cls.g2 = gc.convert(s2)
        cls.x = np.random.normal(size=(10, 4))

    def test_pipe(self):
        pipe = Pipe(layers=[_Layer(), _Layer()])
        y = pipe(self.x)
        self.assertTrue(np.linalg.norm(self.x**4 - y) < 0.001)

    def test_mlp(self):
        layer = MLP(neurons=[10, 3], activations="swish")
        out = layer(self.x)
        self.assertTupleEqual(tuple(out.shape), (10, 3))
        out2 = self.x
        for x in layer.pipe.layers:
            out2 = x(out2)
        np.testing.assert_array_almost_equal(out, out2)

    def test_gated_mlp(self):
        layer = GatedMLP(neurons=[10, 3], activations="swish")
        self.assertTrue(isinstance(layer.pipe, Pipe))
        self.assertTrue(isinstance(layer.gate, Pipe))
        self.assertTrue(layer.gate.layers[-1].activation == tf.keras.activations.sigmoid)

    def test_embedding(self):
        emb = Embedding(2, 8)
        x = np.array([[0], [1], [0], [1]])
        self.assertTupleEqual(tuple(emb(x).shape), (4, 8))


if __name__ == "__main__":
    unittest.main()
