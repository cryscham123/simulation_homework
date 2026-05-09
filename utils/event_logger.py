from typing import Optional
import simpy
import os

class EventLogger:
    def __init__(self, env: simpy.Environment):
        self.__logs = []
        self.__env = env
        
    @property
    def now(self):
        TIME_UNIT = os.getenv('TIME_UNIT', 'M')
        time_constants = {'M': 1, 'H': 60, 'D': 1440}
        return self.__env.now / time_constants[TIME_UNIT]
    
    @property
    def logs(self):
        TIME_UNIT = os.getenv('TIME_UNIT', 'M')
        time_constants = {
            'M': 1,
            'H': 60,
            'D': 60 * 24
        }
        return [
            {
                **log,
                'start': log['start'] / time_constants[TIME_UNIT],
                'finish': log['finish'] / time_constants[TIME_UNIT] if 'finish' in log else None
            }
            for log in self.__logs
        ]

    def log_event_start(self, 
                        id: str, 
                        event: str, 
                        resource: str, 
                        op_id: Optional[int] = None,
                        description: Optional[str] = None):
        self.__logs.append({
            'id': id,
            'event': event,
            'op_id': op_id,
            'description': description,
            'resource': resource,
            'start': self.__env.now
        })
        return len(self.__logs) - 1

    def log_event_finish(self, index: int):
        if index < 0:
            return
        self.__logs[index]['finish'] = self.__env.now
