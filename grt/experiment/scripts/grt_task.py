import json
from asyncio import Event
from asyncio import create_task
from asyncio import wait
from asyncio import wait_for
from asyncio import FIRST_COMPLETED
from asyncio import sleep
from asyncio import TimeoutError

from collections import OrderedDict
from datetime import datetime
from random import randint
from random import shuffle
from time import time
from time import monotonic
from os import environ
from os import path
import math
import sys

import logging

from pprint import pprint as pp

trial_duration = 10

class GRT:
    def __init__(self, participant_no):
        self._start = False
        self._ended = False
        self._participant_no = participant_no

        self._features  = ["Symmetry", "Border", "Colour"]
        self._levels    = ["Low","High"]
        # put this into the state: condition   = conditions[self._participant_no % len(conditions),

        self._design  = dict(
            conditions      = [(self._features[i], self._features[j]) for i in range(len(self._features)) for j in range(i+1, len(self._features))],
            levels          = [[l1, l2] for l1 in self._levels for l2 in self._levels],
            block_trials= 4, #100,
            blocks      = 2
        )

        # this should go into the state, too.
        # self._design["trials"] = [self._design["levels"] * (self._design["block_trials"] // len(self._design["levels"])) for _ in range(self._design["blocks"])]
        # for i in range(self._design["blocks"]):
            # shuffle(self._design["trials"][i])

        self._dim = dict(
            width   = 800, 
            height  = 600,
            img_w   = 256, 
            img_h   = 192, 
            img_sep = 0.1, 
            cue_size= 0.1
        )
        self._pos = dict(
            img_left_x  = (self._dim["width"] // 2) - (self._dim["width"] * self._dim["img_sep"] // 2) - self._dim["img_w"],
            img_right_x = (self._dim["width"] // 2) + (self._dim["width"] * self._dim["img_sep"] // 2),
            img_y       = (self._dim["height"]// 2) - (self._dim["img_h"] // 2),
            cue_x       = (self._dim["width"] // 2) - (self._dim["img_h"] // self._dim["cue_size"] // 2),
            cue_y       = (self._dim["height"]// 2) - (self._dim["img_h"] // self._dim["cue_size"] // 2)
        )
        self.cue = "+", 
        self._timing = dict(
            cue     = 200,
            trial   = 8000,
            isi     = 200,
            block_br= 20_000
        )


a = GRT(1)
pp(a._design)
