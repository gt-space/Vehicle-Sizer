from Layout import Network, Component, State
import numpy as np
from thermoprop import Fluid


class Pipe(Component):

    def __init__(self, name, network, P1, P2, Cd, A, L, mdot):
        self.setup()

    def evaluate(self):
        P1 = self.P1.value
        P2 = self.P2.value
        Cd = self.Cd.value
        A = self.A.value
        L = self.L.value
        mdot = self.mdot.value

        CdA = Cd*A
        R = 1/(2*(CdA**2))
        Z = L/A

        mdot_dot = ((P1-P2) - (R / 1000) * mdot * abs(mdot)) / Z

        self.derivative = mdot_dot

    @property
    def dynamics(self):
        return [(self.mdot, self.derivative)]



class Volume(Component):

    def __init__(self, name, network, P, V, mass = None, mdot_in = None, mdot_out = None):
        self.setup()

    def evaluate(self):
        P = self.P.value
        V = self.V.value

        rho = Fluid("water", pressure=P, temperature=300).density
        self.mass.value = rho*V

    @property
    def dynamics(self):
        return[(self.P, self.mass, self.mdot_in.value - self.mdot_out.value)]




        
Test = Network("Test")

pressure = State(2e5)

Line1 = Pipe("Line 1", Test, 3e5, pressure, 1, (np.pi/4)* (0.5 / 39.37)**2, 3, 0)
Node = Volume(
    "Vol",
    Test,
    P=Line1.P2,
    V=(np.pi/4)*(0.5 / 39.37)**2,
    mdot_in=Line1.mdot,
    mdot_out=0,
)

Line2 = Pipe("Line 2", Test, Node.P, 101325, 1, (np.pi/4)* (0.5 / 39.37)**2, 3, mdot=Node.mdot_out)

Test.evaluate()
print(Test.collect_states())
print(Test.collect_stored_states())
print(Test.collect_derivatives())
print(Test.collect_balances())
