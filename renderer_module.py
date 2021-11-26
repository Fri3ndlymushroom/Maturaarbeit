from rlbot.messages.flat.QuickChatSelection import QuickChatSelection

import random

class Renderer():

    def predictBallPath(self):
        ball_prediction = self.get_ball_prediction_struct()

        loc1 = None
        loc2 = None

        if ball_prediction is not None:
            for i in range(0, ball_prediction.num_slices):
                prediction_slice = ball_prediction.slices[i]
                location = prediction_slice.physics.location

                try:
                    loc1 = ball_prediction.slices[i].physics.location
                    loc2 = ball_prediction.slices[i+6].physics.location
                except:
                    None


                if loc1 is not None and loc2 is not None:
                    if(i%6==0):
                        self.renderer.draw_line_3d(
                            loc1, loc2, self.renderer.yellow())


    def renderText(self, text):
        text = str(text)
        self.renderer.draw_string_2d(4, 4, 2, 2, text, self.renderer.white())



    def chat(self):
        message_index = round(random.random()*6)

        if(self.my_car.score_info.goals > self.goals[0]):
            if(message_index == 0):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Toxic_WasteCPU)
            if(message_index == 1):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_Pro)
            if(message_index == 2):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Exclamation_Yeet)
            if(message_index == 3):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Information_TakeTheShot)
            if(message_index == 4):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Compliments_WhatASave)
            if(message_index == 5):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Reactions_Calculated)
            if(message_index == 6):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Apologies_NoProblem)
            self.goals[0] = self.my_car.score_info.goals
        if(self.packet.game_cars[1].score_info.goals > self.goals[1]):

            if(message_index == 0):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Useful_Bumping)
            if(message_index == 1):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_SkillLevel)
            if(message_index == 2):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Excuses_Lag)
            if(message_index == 3):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Apologies_Whoops)
            if(message_index == 4):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Compliments_Thanks)
            if(message_index == 5):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Reactions_NoWay)
            if(message_index == 6):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_TinyChances)
            self.goals[1] = self.packet.game_cars[1].score_info.goals
