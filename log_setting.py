import logging, colorama
#test
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console
import time
from args import Args

class MyFormatter(logging.Formatter):
    colorama.init(autoreset=True)
    
    RESET = colorama.Style.RESET_ALL
    FORMAT = f"{colorama.Style.BRIGHT}%(levelname)s {colorama.Fore.LIGHTMAGENTA_EX}%(module)s-%(lineno)d: {RESET}"
    MESSAGE = "%(message)s" + RESET

    FORMATS = {
        logging.DEBUG: colorama.Fore.LIGHTGREEN_EX + FORMAT + colorama.Fore.LIGHTGREEN_EX + MESSAGE,
        logging.INFO: colorama.Fore.LIGHTCYAN_EX + FORMAT + colorama.Fore.LIGHTCYAN_EX + MESSAGE,
        logging.WARNING: colorama.Fore.YELLOW +FORMAT + colorama.Fore.YELLOW + MESSAGE,
        logging.ERROR: colorama.Fore.LIGHTRED_EX + FORMAT + colorama.Fore.LIGHTRED_EX + MESSAGE,
        logging.CRITICAL: colorama.Fore.RED + FORMAT + colorama.Fore.RED + MESSAGE
    }
    FORMAT = f"{colorama.Style.BRIGHT}%(levelname)s {colorama.Fore.LIGHTMAGENTA_EX}%(module)s-%(lineno)d: {RESET}"
    FORMATS = {
        logging.DEBUG: colorama.Fore.LIGHTGREEN_EX + FORMAT + colorama.Fore.LIGHTGREEN_EX + MESSAGE,
        logging.INFO: colorama.Fore.LIGHTCYAN_EX + FORMAT + colorama.Fore.LIGHTCYAN_EX + MESSAGE,
        logging.WARNING: colorama.Fore.YELLOW +FORMAT + colorama.Fore.YELLOW + MESSAGE,
        logging.ERROR: colorama.Fore.LIGHTRED_EX + FORMAT + colorama.Fore.LIGHTRED_EX + MESSAGE,
        logging.CRITICAL: colorama.Fore.RED + FORMAT + colorama.Fore.RED + MESSAGE
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


class MyLogging:
    _my_logger = logging.getLogger("my_default_logger")
    _has_mylogger_handler = False
    _root_logger = logging.getLogger()
    _has_root_handler = False

    @classmethod
    def get_default_logger(cls) -> logging.Logger:
        cls._my_logger.setLevel(logging.DEBUG)

        if  not cls._has_mylogger_handler:
            handler = logging.StreamHandler()
            handler.setFormatter(MyFormatter())
            cls._my_logger.addHandler(handler)
            cls._has_mylogger_handler = True

        return cls._my_logger
    
    @classmethod
    def get_root_logger(cls) -> logging.Logger:
        cls._root_logger.setLevel(logging.DEBUG)

        if not cls._has_root_handler:
            handler = logging.StreamHandler()
            handler.setFormatter(MyFormatter())
            cls._root_logger.addHandler(handler)
            cls._has_root_handler = True

        return cls._root_logger

    
    
    

def test_rich():
    from rich import traceback
    traceback.install(show_locals=True)

    logger = MyLogging.get_default_logger()

    console = Console()
    console.print("test1")
    for k in track(range(2), description="outer"):
        for i in track(range(3), transient=True, description="inner"):
            logger.debug("\n+test")
            time.sleep(1)
    
    raise AssertionError


def main():
    logger = MyLogging.get_root_logger()
    root_logger = logging.getLogger()
    print(root_logger.handlers)
    print(logger.handlers)

if __name__ == "__main__":
    main()
    # test_rich()
    # main()

    # from rich import pretty, traceback
    # traceback.install(show_locals=True)
    # pretty.install()
    # logger = logging.getLogger("my_default_logger")
    # logger.setLevel(logging.DEBUG)
    # handler = RichHandler(show_time=False, show_path=False,markup=True, show_level=False)

    # format = f"[bold][spring_green2]%(levelname)s[/spring_green2][/bold] [bold][bright_magenta]%(module)s-%(lineno)s: [/bold][/bright_magenta] %(message)s"
    # date_fmt = f"%Y-%m-%d %H:%M:%S"
    # handler.setFormatter(logging.Formatter(fmt=format, datefmt=date_fmt))
    # logger.addHandler(handler)

    # root_logger = logging.getLogger("temp")
    # root_logger.setLevel(logging.DEBUG)
    # root_handler = logging.StreamHandler()
    # root_handler.setFormatter(MyFormatter())
    # root_logger.addHandler(root_handler)

    # for k in track(range(2),description="outer"):
    #     for i in track(range(3), transient=True, description="inner"):
    #         logger.warning(Args.COACH_ARGS)
    #         # root_logger.debug("testing")
    #         time.sleep(0.5)
    
    # root_logger.debug(Args.COACH_ARGS)
    # raise AssertionError("test")