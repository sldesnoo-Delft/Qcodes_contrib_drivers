import threading
import queue
import logging
from typing import Dict, Callable, Any

class Worker:
    '''
    Asynchronous executor of arbitrary calls.
    '''

    __ACTION_STOP = 'stop_worker_thread'

    _modules: Dict[str, 'Worker'] = {}
    """ All workers by name. """

    @classmethod
    def get_worker(cls, name:str):
        if name in cls._modules:
            return cls._modules[name]
        else:
            return Worker(name)


    def __init__(self, name):
        if name in self._modules:
            raise Exception(f'Worker with name {name} already active')
        self.name = name
        self._modules[name] = self

        self._users = []
        self._action_queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=self.name)
        self._thread.start()

    def _close(self):
        del self._modules[self.name]
        self._action_queue.put(self.__ACTION_STOP)
        # wait at most 15 seconds. Should be more enough for normal scenarios
        self._thread.join(15)
        if self._thread.is_alive():
            logging.error(f'Worker thread {self.name} stop failed. Thread still running.')
        self._action_queue = None
        self._thread = None

    def start(self, user:str):
        self._users.append(user)

    def stop(self, user:str):
        self._users.remove(user)
        if len(self._users) == 0:
            self._close()

    def execute_async(self, call:Callable[..., Any]):
        self._action_queue.put(call)

    def _run(self):
        logging.info('Uploader starting')

        while True:
            entry = self._action_queue.get()

            if entry == Worker.__ACTION_STOP:
                break

            entry()

        logging.info('Worker terminated')

