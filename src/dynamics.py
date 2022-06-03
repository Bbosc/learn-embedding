#!/usr/bin/env python

import torch
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(self, attractor, stiffness, dissipation, embedding):
        super(Dynamics, self).__init__()

        self.attractor = attractor

        self.stiffness = stiffness

        self.dissipation = dissipation

        self.embedding = embedding

    # Forward network pass
    def forward(self, X):
        # data
        x = X[:, :int(X.shape[1]/2)]
        v = X[:, int(X.shape[1]/2):]

        # embedding
        f = self.embedding(x)

        # jacobian
        j = self.embedding.jacobian(x, f)

        # metric
        m = self.embedding.pullmetric(j)

        # christoffel
        g = self.embedding.christoffel(x, m)

        return (torch.bmm(m.inverse(), -(self.dissipation(v)+self.stiffness(x-self.attractor)).unsqueeze(2)) - torch.bmm(torch.einsum('bqij,bi->bqj', g, v), v.unsqueeze(2))).squeeze()

    # Potential function
    def potential(self, x):
        d = x - self.attractor

        return (d*self.stiffness(d)).sum(axis=1)

    # Attractor setter/getter
    @property
    def attractor(self):
        return self.attractor_

    @attractor.setter
    def attractor(self, value):
        self.attractor_ = value

    # Stiffness matrix setter/getter
    @property
    def stiffness(self):
        return self.stiffness_

    @stiffness.setter
    def stiffness(self, value):
        self.stiffness_ = value

    # Dissipative matrix setter/getter
    @property
    def dissipation(self):
        return self.dissipation_

    @dissipation.setter
    def dissipation(self, value):
        self.dissipation_ = value

    # Diffeomorphism setter/getter
    @property
    def embedding(self):
        return self.embedding_

    @embedding.setter
    def embedding(self, value):
        self.embedding_ = value
