import abc

from typing import Any

from forager_knn.utils import JSONType


class Reducer(abc.ABC):
    def handle_chunk_result(self, chunk: JSONType, chunk_output: JSONType) -> None:
        pass

    def finish(self) -> None:
        pass

    @abc.abstractmethod
    def handle_result(self, input: JSONType, output: JSONType) -> None:
        pass

    @abc.abstractproperty
    def result(self) -> Any:
        pass
