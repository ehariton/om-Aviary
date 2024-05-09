import unittest
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilderBase
from motor_builder import MotorBuilder
from motor_variables import Aircraft
from aviary.utils.aviary_values import AviaryValues


class TestMotor(TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = MotorBuilder()
        self.aviary_values = AviaryValues()
        self.aviary_values.set_val(Aircraft.Gearbox.GEAR_RATIO, 1)


if __name__ == '__main__':
    unittest.main()
