from util.sequence import Sequence, ControlStep

from rlbot.agents.base_agent import SimpleControllerState

class Sequence():

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.02,
                        controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.02,
                        controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(
                jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def begin_speed_flip(self, packet):
        # Send some quickchat just for fun

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(
                duration=0.7, controls=SimpleControllerState(boost=True)),
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.2, controls=SimpleControllerState(
                jump=False, yaw=-1, pitch=-1)),
            ControlStep(duration=0.1, controls=SimpleControllerState(
                jump=True, yaw=-1, pitch=-1, )),
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(yaw=-1, pitch=1)),
            ControlStep(duration=0.8, controls=SimpleControllerState(yaw=0.5)),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)