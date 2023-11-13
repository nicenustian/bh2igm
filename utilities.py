#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:07:30 2023

@author: nasir
"""

class UtilityFunctions:
    # Math operations
    def square(self, x):
        return x ** 2

    def cube(self, x):
        return x ** 3

    # String manipulation
    def reverse_string(self, text):
        return text[::-1]

    def capitalize(self, text):
        return text.upper()

    # File handling
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def write_file(self, file_path, content):
        with open(file_path, 'w') as file:
            file.write(content)
