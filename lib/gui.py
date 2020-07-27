import pygame
from pygame.locals import *

from .stdlib.collections import dotdict


class GUI():
    prefabs = dotdict()
    color = dotdict()
    font = dotdict()

    _current_line_position = 0

    @classmethod
    def init(cls):
        cls.color.white = (0xFF, 0xFF, 0xFF)
        cls.color.black = (0x00, 0x00, 0x00)
        cls.color.red = (0xFF, 0x00, 0x00)
        cls.color.green = (0x00, 0xFF, 0x00)
        cls.color.blue = (0x00, 0x00, 0xFF)
        cls.color.font_normal = (0xFF, 0xFF, 0xFF)
        cls.color.screen_backgorund = (0x1F, 0x1F, 0x1F)

        cls.font.normal = pygame.font.Font(
            "fonts/Ubuntu Mono derivative Powerline.ttf", 20)
        cls.font.large = pygame.font.Font(
            "fonts/Ubuntu Mono derivative Powerline.ttf", 24)  # x 1.2
        cls.font.xlarge = pygame.font.Font(
            "fonts/Ubuntu Mono derivative Powerline.ttf", 30)  # x 1.5

    @classmethod
    def make_text(cls, text, font=None, color=None):
        if font is None:
            font = cls.font.normal
        if color is None:
            color = cls.color.font_normal

        return font.render(text, True, color)

    @classmethod
    def draw_multiline_text(cls, screen, text, color=None):
        if color is None:
            color = cls.color.font_normal
        x, y = cls._current_line_position
        for line in text.splitlines():
            screen.blit(cls.make_text(line, color=color), (x, y))
            y += 20
        cls._current_line_position = (x, y)

    @classmethod
    def begin_multilines(cls, position):
        cls._current_line_position = position
