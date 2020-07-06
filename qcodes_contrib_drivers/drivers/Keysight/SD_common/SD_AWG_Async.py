# -*- coding: utf-8 -*-
import threading
from typing import Dict, List, Union, Optional, TypeVar, Callable, Any
import sys
import time
import logging
from functools import wraps

import numpy as np

from .SD_Module import keysightSD1, result_parser
from .SD_AWG import SD_AWG
from .memory_manager import MemoryManager
from .workers import Worker


F = TypeVar('F', bound=Callable[..., Any])

def switchable(switch, enabled:bool) -> Callable[[F], F]:
    '''
    This decorator enables or disables a method depending on the value of an object's method.
    It throws an exception when the invoked method is disabled.
    '''

    def switchable_decorator(func):

        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            if switch(self) != enabled:
                switch_name = f'{type(self).__name__}.{switch.__name__}()'
                raise Exception(f'{func.__name__} is not enabled when {switch_name} == {switch(self)}')
            return func(self, *args, **kwargs)


        return func_wrapper

    return switchable_decorator


class WaveformReference:
    """
    This is a reference to a waveform (being) uploaded to the AWG.

    Attributes:
        wave_number: number refering to the wave in AWG memory
        awg_name: name of the awg the waveform is uploaded to
    """
    def __init__(self, wave_number: int, awg_name: str):
        self._wave_number = wave_number
        self._awg_name = awg_name

    @property
    def wave_number(self):
        """
        Number of the wave in AWG memory.
        """
        return self._wave_number

    @property
    def awg_name(self):
        """
        Name of the AWG the waveform is uploaded to
        """
        return self._awg_name

    def release(self):
        """
        Releases the AWG memory for reuse.
        """
        raise NotImplementedError()

    def wait_uploaded(self):
        """
        Waits till waveform has been loaded.
        Returns immediately if waveform is already uploaded.
        """
        raise NotImplementedError()

    def is_uploaded(self):
        """
        Returns True if waveform has been loaded.
        """
        raise NotImplementedError()


class _WaveformReferenceInternal(WaveformReference):

    def __init__(self, allocated_slot: MemoryManager.AllocatedSlot, awg_name: str):
        super().__init__(allocated_slot.number, awg_name)
        self._allocated_slot = allocated_slot
        self._uploaded = threading.Event()
        self._upload_error: Optional[str] = None
        self._released: bool = False
        self._queued_count = 0


    def release(self):
        """
        Releases the memory for reuse.
        """
        if self._released:
            raise Exception('Reference already released')

        self._released = True
        self._try_release_slot()


    def wait_uploaded(self):
        """
        Waits till waveform is loaded.
        Returns immediately if waveform is already uploaded.
        """
        if self._released:
            raise Exception('Reference already released')

        # complete memory of AWG can be written in ~ 15 seconds
        ready = self._uploaded.wait(timeout=30.0)
        if not ready:
            raise Exception(f'Timeout loading wave')

        if self._upload_error:
            raise Exception(f'Error loading wave: {self._upload_error}')


    def is_uploaded(self):
        """
        Returns True if waveform has been loaded.
        """
        if self._upload_error:
            raise Exception(f'Error loading wave: {self._upload_error}')

        return self._uploaded.is_set()


    def enqueued(self):
        self._queued_count += 1


    def dequeued(self):
        self._queued_count -= 1
        self._try_release_slot()


    def _try_release_slot(self):
        if self._released and self._queued_count <= 0:
            self._allocated_slot.release()


    def __del__(self):
        if not self._released:
            logging.warn(f'WaveformReference was not released ({self.awg_name}:{self.wave_number}). '
                         'Automatic release in destructor.')
            self.release()


class SD_AWG_Async(SD_AWG):
    """
    # TODO @@@ update description
    Generic asynchronous driver with waveform memory management for Keysight SD AWG modules.

    This driver is derived from SD_AWG and uses a thread to upload waveforms.
    This class creates reusable memory slots of different sizes in AWG.
    It assigns waveforms to the smallest available memory slot.

    Only one instance of this class per AWG module is allowed.
    By default the maximum size of a waveform is limited to 1e6 samples.
    This limit can be increased up to 1e8 samples at the cost of a longer startup time of the threads.

    The memory manager and asynchronous functionality can be disabled to restore the behavior of
    the parent class. The instrument can then be used with old synchronous code.

    Example:
        awg1 = SW_AWG_Async('awg1', 0, 1, channels=4, triggers=8)
        awg2 = SW_AWG_Async('awg2', 0, 2, channels=4, triggers=8)
        awg3 = SW_AWG_Async('awg3', 0, 3, channels=4, triggers=8)

        # the upload to the 3 modules will run concurrently (in background)
        ref_1 = awg1.upload_waveform(wave1)
        ref_2 = awg2.upload_waveform(wave2)
        ref_3 = awg3.upload_waveform(wave3)

        trigger_mode = keysightSD1.SD_TriggerModes.EXTTRIG
        # method awg_queue_waveform blocks until reference waveform has been uploaded.
        awg1.awg_queue_waveform(1, ref_1, trigger_mode, 0, 1, 0)
        awg2.awg_queue_waveform(1, ref_2, trigger_mode, 0, 1, 0)
        awg3.awg_queue_waveform(1, ref_3, trigger_mode, 0, 1, 0)

    Args:
        name (str): an identifier for this instrument, particularly for
            attaching it to a Station.
        chassis (int): identification of the chassis.
        slot (int): slot of the module in the chassis.
        channels (int): number of channels of the module.
        triggers (int): number of triggers of the module.
        legacy_channel_numbering (bool): indicates whether legacy channel number
            should be used. (Legacy numbering starts with channel 0)
        waveform_size_limit (int): maximum size of waveform that can be uploaded
        asynchronous (bool): if False the memory manager and asynchronous functionality are disabled.
    """

    _modules: Dict[str, 'SD_AWG_Async'] = {}
    """ All async modules by unique module id. """

    MODE_BYPASS = 0
    ''' Bypass memory management and asynchronous functionality; effectively pass all calls to SD_AWG. '''

    MODE_SYNCHRONOUS = 1
    ''' Use memory management with synchronous (blocking) waveform upload.'''

    MODE_SINGLE_THREAD = 2
    ''' Use memory management and 1 shared uploader thread for all modules.'''

    MODE_MULTI_THREADED = 3
    ''' Use waveform memory management and an uploader thread for each module.'''

    def __init__(self, name, chassis, slot, channels, triggers, waveform_size_limit=1e6,
                 mode=MODE_SINGLE_THREAD, **kwargs):
        super().__init__(name, chassis, slot, channels, triggers, **kwargs)

        self._waveform_size_limit = waveform_size_limit

        module_id = self._get_module_id()
        if module_id in SD_AWG_Async._modules:
            raise Exception(f'AWG module {module_id} already exists')

        self.module_id = module_id
        SD_AWG_Async._modules[self.module_id] = self

        self._mode = self.MODE_BYPASS
        self._async_uploader = None
        self.set_mode(mode)


    def memory_managed(self):
        return self._mode != self.MODEBYPASS


    def set_mode(self, mode):
        """
        Sets the memory management and asynchronous mode of the SD_AWG_Async object.
        """
        if mode == self._mode:
            return

        if self._async_uploader:
            self._async_uploader.stop(self)
            self._async_uploader = None

        self._stop_memory_manager()

        self._mode = mode

        if mode == self.MODE_BYPASS:
            return

        self._start_memory_manager()

        if mode == self.MODE_SINGLE_THREAD:
            self._async_uploader = Worker.get_worker('SingleThread')
            self._async_uploader.start(self.name)
        elif mode == self.MODE_MULTI_THREADED:
            self._async_uploader = Worker.get_worker(self.name)
            self._async_uploader.start(self.name)

        slots = self._memory_manager.get_uninitialized_slots()

        if mode == self.MODE_SYNCHRONOUS:
            self.__init_awg_memory(slots, flush_memory=True)
        else:
            self._async_uploader.execute_async(lambda : self.__init_awg_memory(slots, flush_memory=True))

    #
    # disable synchronous method of parent class, when wave memory is managed by this class.
    #
    @switchable(memory_managed, enabled=False)
    def load_waveform(self, waveform_object, waveform_number, verbose=False):
        super().load_waveform(waveform_object, waveform_number, verbose)

    @switchable(memory_managed, enabled=False)
    def load_waveform_int16(self, waveform_type, data_raw, waveform_number, verbose=False):
        super().load_waveform_int16(waveform_type, data_raw, waveform_number, verbose)

    @switchable(memory_managed, enabled=False)
    def reload_waveform(self, waveform_object, waveform_number, padding_mode=0, verbose=False):
        super().reload_waveform(waveform_object, waveform_number, padding_mode, verbose)

    @switchable(memory_managed, enabled=False)
    def reload_waveform_int16(self, waveform_type, data_raw, waveform_number, padding_mode=0, verbose=False):
        super().reload_waveform_int16(waveform_type, data_raw, waveform_number, padding_mode, verbose)

    @switchable(memory_managed, enabled=False)
    def flush_waveform(self, verbose=False):
        super().flush_waveform(verbose)

    @switchable(memory_managed, enabled=False)
    def awg_from_file(self, awg_number, waveform_file, trigger_mode, start_delay, cycles, prescaler, padding_mode=0,
                      verbose=False):
        super().awg_from_file(awg_number, waveform_file, trigger_mode, start_delay, cycles, prescaler, padding_mode,
                              verbose)

    @switchable(memory_managed, enabled=False)
    def awg_from_array(self, awg_number, trigger_mode, start_delay, cycles, prescaler, waveform_type, waveform_data_a,
                       waveform_data_b=None, padding_mode=0, verbose=False):
        super().awg_from_array(awg_number, trigger_mode, start_delay, cycles, prescaler, waveform_type, waveform_data_a,
                       waveform_data_b, padding_mode, verbose)


    def awg_flush(self, awg_number):
        super().awg_flush(awg_number)
        if self.memory_managed():
            self._release_waverefs_awg(awg_number)


    def awg_queue_waveform(self, awg_number, waveform_ref, trigger_mode, start_delay, cycles, prescaler):
        """
        Enqueus the waveform.

        Args:
            awg_number (int): awg number (channel) where the waveform is queued
            waveform_ref (WaveformReference): reference to a waveform
            trigger_mode (int): trigger method to launch the waveform
                Auto                        :   0
                Software/HVI                :   1
                Software/HVI (per cycle)    :   5
                External trigger            :   2
                External trigger (per cycle):   6
            start_delay (int): defines the delay between trigger and wf launch
                given in multiples of 10ns.
            cycles (int): number of times the waveform is repeated once launched
                zero = infinite repeats
            prescaler (int): waveform prescaler value, to reduce eff. sampling rate
        """
        if self.memory_managed():
            if waveform_ref.awg_name != self.name:
                raise Exception(f'Waveform not uploaded to this AWG ({self.name}). '
                                f'It is uploaded to {waveform_ref.awg_name}')

            self.log.debug(f'Enqueue {waveform_ref.wave_number}')
            if not waveform_ref.is_uploaded():
                start = time.perf_counter()
                self.log.debug(f'Waiting till wave {waveform_ref.wave_number} is uploaded')
                waveform_ref.wait_uploaded()
                duration = time.perf_counter() - start
                self.log.info(f'Waited {duration*1000:5.1f} ms for upload of wave {waveform_ref.wave_number}')

            waveform_ref.enqueued()
            self._enqueued_waverefs[awg_number].append(waveform_ref)
            wave_number = waveform_ref.wave_number
        else:
            wave_number = waveform_ref

        super().awg_queue_waveform(awg_number, wave_number, trigger_mode, start_delay, cycles, prescaler)


    @switchable(memory_managed, enabled=True)
    def set_waveform_limit(self, requested_waveform_size_limit: int):
        """
        Increases the maximum size of waveforms that can be uploaded.

        Additional memory will be reserved in the AWG.
        Limit can not be reduced, because reservation cannot be undone.

        Args:
            requested_waveform_size_limit (int): maximum size of waveform that can be uploaded
        """
        self._memory_manager.set_waveform_limit(requested_waveform_size_limit)
        slots = self._memory_manager.get_uninitialized_slots()

        if self._mode == self.MODE_SYNCHRONOUS:
            self.__init_awg_memory(slots, flush_memory=False)
        else:
            self._async_uploader.execute_async(lambda : self.__init_awg_memory(slots, flush_memory=False))


    @switchable(memory_managed, enabled=True)
    def reinitialize_waveform_memory(self):
        self._stop_memory_manager()

        self._start_memory_manager()

        slots = self._memory_manager.get_uninitialized_slots()

        if self._mode == self.MODE_SYNCHRONOUS:
            self.__init_awg_memory(slots, flush_memory=True)
        else:
            self._async_uploader.execute_async(lambda : self.__init_awg_memory(slots, flush_memory=True))


    @switchable(memory_managed, enabled=True)
    def upload_waveform(self, wave: Union[List[float], List[int], np.ndarray]) -> WaveformReference:
        '''
        Upload the wave using the uploader thread for this AWG.
        Args:
            wave: wave data to upload.
        Returns:
            reference to the wave
        '''
        if len(wave) < 2000:
            raise Exception(f'{len(wave)} is less than 2000 samples required for proper functioning of AWG')

        allocated_slot = self._memory_manager.allocate(len(wave))
        ref = _WaveformReferenceInternal(allocated_slot, self.name)
        self.log.debug(f'upload: {ref.wave_number}')
        if self._mode == self.MODE_SYNCHRONOUS:
            self.__upload(ref, wave)
        else:
            self._async_uploader.execute_async(lambda : self.__upload(ref, wave))

        return ref


    def close(self):
        """
        Closes the module and stops background thread.
        """
        self.log.info(f'stopping ({self.module_id})')
        self._stop_memory_manager()
        if self._async_uploader:
            self._async_uploader.stop(self.name)
            self._async_uploader = None

        del SD_AWG_Async._modules[self.module_id]

        super().close()


    def _get_module_id(self):
        """
        Generates a unique name for this module.
        """
        return f'{self.module_name}:{self.chassis_number()}-{self.slot_number()}'


    def _start_memory_manager(self):
        """
        Starts the memory manager.
        """
        self._memory_manager = MemoryManager(self.log, self._waveform_size_limit)
        self._enqueued_waverefs = {}
        for i in range(self.channels):
            self._enqueued_waverefs[i+1] = []


    def _stop_memory_manager(self):
        """
        Stops the memory manager.
        """
        if self.memory_managed():
            self._release_waverefs()
            self._memory_manager = None


    def _release_waverefs(self):
        for i in range(self.channels):
            self._release_waverefs_awg(i + 1)


    def _release_waverefs_awg(self, awg_number):
        for waveref in self._enqueued_waverefs[awg_number]:
            waveref.dequeued()
        self._enqueued_waverefs[awg_number] = []


    ### Following methods may be called from a different thread context

    def __init_awg_memory(self, new_slots:List[MemoryManager.MemorySlot], flush_memory:bool=False):
        '''
        Initialize memory on the AWG by uploading waveforms with all zeros.
        '''
        if flush_memory:
            # invoke flush on synchronous parent class
            super().flush_waveform()

        if len(new_slots) == 0:
            return

        self.log.debug(f'Reserving awg memory for {len(new_slots)} slots on awg {self.name}')

        zeros = []
        total_size = 0
        total_duration = 0
        for slot in new_slots:
            start = time.perf_counter()
            if len(zeros) != slot.size:
                zeros = np.zeros(slot.size, np.float)
                wave = keysightSD1.SD_Wave()
                result_parser(wave.newFromArrayDouble(keysightSD1.SD_WaveformTypes.WAVE_ANALOG, zeros))
            # invoke load_waveform on synchronous parent class
            super().load_waveform(wave, slot.number)
            duration = time.perf_counter() - start
            total_duration += duration
            total_size +=  slot.size
            self.log.debug(f'uploaded {slot.size} in {duration*1000:5.2f} ms ({slot.size/duration/1e6:5.2f} MSa/s)')

        self.log.info(f'Awg memory reserved: {len(new_slots)} slots, {total_size/1e6} MSa in '
                      f'{total_duration*1000:5.2f} ms ({total_size/total_duration/1e6:5.2f} MSa/s) on awg {self.name}')

    def __upload(self,
                 wave_ref: WaveformReference,
                 wave_data: Union[List[float], List[int], np.ndarray]):

            self.log.debug(f'Uploading {wave_ref.wave_number} on awg {self.name}')
            try:
                start = time.perf_counter()

                sd_wave = keysightSD1.SD_Wave()
                result_parser(sd_wave.newFromArrayDouble(keysightSD1.SD_WaveformTypes.WAVE_ANALOG, wave_data))
                # invoke reload_waveform on synchronous parent class
                super().reload_waveform(sd_wave, wave_ref.wave_number)

                duration = time.perf_counter() - start
                speed = len(wave_data)/duration
                self.log.debug(f'Uploaded {wave_ref.wave_number} in {duration*1000:5.2f} ms ({speed/1e6:5.2f} MSa/s)')
            except:
                ex = sys.exc_info()
                msg = f'{ex[0].__name__}:{ex[1]}'
                min_value = np.min(wave_data)
                max_value = np.max(wave_data)
                if min_value < -1.0 or max_value > 1.0:
                    msg += ': Voltage out of range'
                self.log.error(f'Failure load waveform {wave_ref.wave_number} on awg {self.name}: {msg}' )
                wave_ref._upload_error = msg

            # signal upload done, either successful or with error
            wave_ref._uploaded.set()


