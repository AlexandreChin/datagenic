class bcolors:
    HEADER = "\033[95m"
    DAY = "\033[94m"
    TIME = "\033[96m"
    INFO = "\033[93m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[31m"


def print_message(txt, logger=False, type="info", where="", break_line=False, console_print=True):

    # terminal_width = os.get_terminal_size().columns
    if console_print:
        if logger:
            if break_line:
                print("\n")
            if type == "important":
                dieses = (len(txt) + 13) * "-"
                print(f"\n{bcolors.DAY}{dieses}")
                print(f"----- {txt} ------")
                print(f"{dieses}{bcolors.ENDC}\n")
                logger.warning(txt)

            elif type == "timer":
                msg = f"{bcolors.TIME}[Timing] : {txt}"
                msg_time = f"{where}{bcolors.ENDC}"
                # msg_time = msg_time.rjust(terminal_width-len(msg)+7)
                print(msg + msg_time + "\n")
                logger.info(txt + where)
            elif type == "info":
                msg = f"{bcolors.BOLD}[Info]{where} :   {txt}{bcolors.ENDC}"
                print(msg)
                logger.info(txt)
            elif type == "final":
                msg = f"{bcolors.GREEN}[Final]{where} : {txt}{bcolors.ENDC}"
                print(msg)
                logger.info(txt)
            elif type == "warning":
                msg = f"{bcolors.RED}[Warning]{where} : {txt}{bcolors.ENDC}"
                print(msg)
                logger.info(txt)
        else:
            if break_line:
                print("\n")
            if type == "important":
                dieses = (len(txt) + 13) * "-"
                print(f"\n{bcolors.DAY}{dieses}")
                print(f"----- {txt} ------")
                print(f"{dieses}{bcolors.ENDC}\n")
            elif type == "timer":
                msg = f"{bcolors.TIME}[Timing] : {txt}"
                msg_time = f"{where}{bcolors.ENDC}"
                # msg_time = msg_time.rjust(terminal_width-len(msg)+7)
                print(msg + msg_time + "\n")
            elif type == "info":
                msg = f"{bcolors.BOLD}[Info]{where} :   {txt}{bcolors.ENDC}"
                print(msg)
            elif type == "final":
                msg = f"{bcolors.GREEN}[Final]{where} : {txt}{bcolors.ENDC}"
                print(msg)
            elif type == "warning":
                msg = f"{bcolors.RED}[Warning]{where} : {txt}{bcolors.ENDC}"
                print(msg)
    else:
        if logger:
            if type == "important":
                logger.warning(txt)
            elif type == "timer":
                logger.info(txt + where)
            elif type == "info" or type == "final":
                logger.info(txt)
            elif type == "warning":
                logger.warning(txt)
