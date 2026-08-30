from Layout import Network, Component, State, Vehicle, Section, Sizer
import numpy as np
from thermoprop import Fluid
from fullplot import Trace, plot


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

    def __init__(self, name, network, P, V, mass = None, mdot_in = None, mdot_out = None, sizer:Sizer = None):
        self.setup()

    def evaluate(self):
        P = self.P.value
        V = self.V.value

        rho = Fluid("water", pressure=P, temperature=300).density
        self.mass.value = rho*V

    @property
    def dynamics(self):
        return[(self.P, self.mass, self.mdot_in.value - self.mdot_out.value)]


class VolumeSizer(Sizer):

    def __init__(self, volume):
        self.volume = volume

    def size(self, vol):
        vol.V.value = self.volume



class TankSection(Section):

    def __init__(self, name, vehicle, tank:Volume):
        self.setup()



Elytra = Vehicle("Elytra")
PropSystem = Network("PropSystem", Elytra)

pressure = State(101325)

vol_sizer = VolumeSizer((np.pi/4)* (1.5 / 39.37)**2)

Line1 = Pipe("Line 1", PropSystem, 3e5, pressure, 1, (np.pi/4)* (0.5 / 39.37)**2, 3, 0)
Node = Volume("Vol", PropSystem, P=Line1.P2, V=3, mdot_in=Line1.mdot,mdot_out=0, sizer=vol_sizer)
Line2 = Pipe("Line 2", PropSystem, Node.P, 101325, 1, (np.pi/4)* (0.5 / 39.37)**2, 3, mdot=Node.mdot_out)

Elytra.size()
sol = Elytra.fly(dt=0.0005, t_final=0.1)

pressure_trace = Trace(
    x=sol["time"],
    y=sol["Vol"]["P"],
    name="Volume pressure",
)

plot(
    [pressure_trace],
    xlabel="Time (s)",
    ylabel="Pressure (Pa)",
    title="Volume Pressure",
    show=True,
)

