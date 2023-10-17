from qcodes.instrument import VisaInstrument
from qcodes.parameters import (DelegateGroup, DelegateGroupParameter,
                               GroupedParameter, Parameter,
                               create_on_off_val_mapping)
from qcodes.validators import Enum


def _delegate_group_factory(name: str, *params: Parameter) -> DelegateGroup:
    """Returns a DelegateGroup with DelegateGroupParameters
    that simply forward the underlying parameter.

    GroupedParameter for some reason requires DelegateGroups.
    """
    return DelegateGroup(
        name,
        [DelegateGroupParameter(param.name, param) for param in params]
    )

class LighthousePhotonicsSproutG(VisaInstrument):
    """Qcodes driver for the Lighthouse Photonics Sprout-G laser.

    Inspired by pylablib.
    """

    def __init__(self, name, address, timeout=5, terminator='\r', device_clear=True, visalib=None,
                 **kwargs):
        super().__init__(name, address, timeout, terminator, device_clear, visalib, **kwargs)

        self.product = Parameter('product', get_cmd='PRODUCT?', instrument=self)
        self.version = Parameter('version', get_cmd='VERSION?', instrument=self)
        self.serial = Parameter('serial', get_cmd='SERIALNUMBER?', instrument=self)
        self.config = Parameter('config', get_cmd='CONFIG?', instrument=self)
        self.device_info = GroupedParameter(
            'device_info',
            group=_delegate_group_factory(
                'device_info',
                self.product, self.version, self.serial, self.config
            ),
            instrument=self
        )
        """The device info."""

        self.on_hours = Parameter('on_hours', get_cmd='HOURS?', unit='hours', instrument=self)
        self.run_hours = Parameter('run_hours', get_cmd='RUN HOURS?', unit='hours',
                                   instrument=self)
        self.work_hours = GroupedParameter(
            'work_hours',
            group=_delegate_group_factory(
                'work_hours',
                self.on_hours, self.run_hours
            ),
            instrument=self
        )
        """The running hours of the controller and the laser."""

        self.output_mode = Parameter('output_mode',
                                     get_cmd='OPMODE?', set_cmd='OPMODE={}',
                                     get_parser=str.lower, set_parser=str.upper,
                                     vals=Enum('on', 'off', 'idle', 'calibrate'),
                                     instrument=self)
        """The output status."""
        self.warning_status = Parameter('warning_status', get_cmd='WARNING?', instrument=self)
        self.shutter_status = Parameter('shutter_status', get_cmd='SHUTTER?', instrument=self)
        self.interlock_status = Parameter('interlock_status', get_cmd='INTERLOCK?',
                                          instrument=self)
        self.status = GroupedParameter(
            'status',
            group=_delegate_group_factory(
                'status',
                self.output_mode, self.warning_status, self.shutter_status,
                self.interlock_status
            ),
            instrument=self
        )
        """Status messages for output mode, warnings, shutter, and interlock."""

        self.enabled = Parameter('enabled',
                                 get_cmd=self.output_mode, set_cmd=self.output_mode,
                                 val_mapping=create_on_off_val_mapping('on', 'off'),
                                 instrument=self)
        """Enable/disable the output."""
        self.output_power = Parameter('output_power', get_cmd='POWER?',
                                      set_cmd='POWER SET={:.2f}',
                                      unit='Watt', instrument=self)
        """The current output power. The setter uses the output_setpoint parameter."""
        self.output_setpoint = Parameter('output_setpoint', get_cmd='POWER SET?', unit='Watt',
                                         instrument=self)
        """The output power setpoint."""

        self.connect_message()

    def ask(self, cmd: str) -> str:
        response = super().ask(cmd)
        if response.startswith(cmd.replace('?', '=')):
            return response[len(cmd):]
        elif response.isnumeric():
            msg = f'Unknown command {cmd}.'
        else:
            msg = f"Unexpected response for command {cmd}: {response}"
        self.visa_log.exception(msg)
        raise RuntimeError(msg)

    def write(self, cmd: str) -> None:
        super().write(cmd)
        # Flush
        response = self.visa_handle.read()
        if response == '0':
            return
        elif response.isnumeric():
            msg = f'Command {cmd} not accepted.'
        else:
            msg = f'Unexpected response for command {cmd}: {response}'
        self.visa_log.exception(msg)
        raise RuntimeError(msg)

    def get_idn(self) -> dict:
        return {'vendor': 'Lighthouse Photonics',
                'model': self.product(),
                'serial': self.serial(),
                'firmware': self.version()}
