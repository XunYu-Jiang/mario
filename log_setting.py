import logging, colorama

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

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


class MyLogging:
    _logger = logging.getLogger("my_default_logger")

    @classmethod
    def get_default_logger(cls) -> logging.Logger:
        handler = logging.StreamHandler()
        handler.setFormatter(MyFormatter())
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)

        return cls._logger