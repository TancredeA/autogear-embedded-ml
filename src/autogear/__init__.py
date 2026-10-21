"""AutoGear extension: learning to shift without a torque sensor.

Sub-modules
-----------
data         download the public sources and rebuild the 1-Hz dataset the prototype would have recorded
gears        recover the engaged gear from the speed-to-cadence ratio (Viterbi)
features     the 14 inputs the device can compute, none of which depends on the gear chosen
controller   the hysteresis controller that turns a target development into one-step shifts
replay       closed-loop replay and the five measures of a policy
policies     decision tables (R0, R1, RT), target models (GBDT, ...), and the event classifier
protocol     within-rider tests and the nested selection of decision parameters
export_c     export of the boosted trees to plain C arrays for the microcontroller
"""
__version__ = "1.0.0"
