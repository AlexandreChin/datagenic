import contextlib
import inspect
import json
import numbers

from requests.packages import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_default_args(func):
    """
    returns the default arguments of a function
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def load_json(json_full_path):
    """
    reads a json file as a dictionary
    """
    try:
        with open(json_full_path) as f:
            data = json.load(f)
        return data
    except OSError:
        print(f"Cant read {json_full_path}")


def load_csv(csv_full_path):
    """
    reads a csv file as a dictionary
    """
    try:
        with open(csv_full_path, "rb") as data:
            pass
        return data
    except OSError:
        print(f"Cant read {csv_full_path}")


def convert_string_to_list_of_dicts(obj):
    """
    Converts string-response to a list of dicts
    """
    # format response
    if type(obj) is str:
        if "\n" in obj:
            # add comma after each json string and convert them to list
            cleaned_str = obj.replace("\n", ",\n").split(",\n")

            # remove empty items from the list
            result = [s for s in cleaned_str if s]

            # convert to json object
            result = json.loads(json.dumps(result))
            result = [json.loads(element) for element in result]
        else:
            result = obj

    elif type(obj) is dict:
        result = [obj]

    return result


def deep_get(dictionary, string_keys):
    """
    get value in a nested dict
    """
    # keys=list(map(lambda x: int(x) if (x.isnumeric()) else x, keys.split(".")))
    keys_list = string_keys.split(".")

    for key in keys_list:
        with contextlib.suppress(BaseException):
            key = int(key)
        dictionary = dictionary[key]

    return dictionary


def deep_get_assign(dictionary, string_keys, value):
    """
    assign value to dictionary[key1]...[keyn]
    """
    keys_list = string_keys.split(".")
    depth = len(keys_list)

    temp = dictionary

    for idx, key in enumerate(keys_list):
        if key.isnumeric():
            key = int(key)
        if idx < depth - 1:
            temp = temp[key]
        else:
            temp[key] = value

    return dictionary


def print_logs(obj, log_full_path):
    """
    obj=object to print (dictionnary,list etc.)
    log_full_path= full path to log file
    """
    with open(log_full_path, "a") as log_file:
        log_file.write(json.dumps(obj, sort_keys=False, indent=4))


def compare_list_of_dicts(query_output, expected_output):
    """
    Compares 2 unordered lists of dictionaries
    """

    trigger = True
    if len(expected_output) == len(query_output):
        if len(expected_output):
            for record in expected_output:
                if record in query_output:
                    pass
                else:
                    trigger = False
                    break
    else:
        trigger = False
    return trigger


def rounding_numerical_values_in_an_obj(obj, num_of_decimals=2):
    """
    Round all numeric values in a object
    """
    if isinstance(obj, list):
        for record in obj:
            for key in record:
                if isinstance(record[key], numbers.Number):
                    record[key] = round(record[key], num_of_decimals)
    elif isinstance(obj, dict):
        for key in obj:
            if isinstance(obj[key], numbers.Number):
                obj[key] = round(obj[key], num_of_decimals)
    else:
        print(f"object type {type(obj)} not supported; please debug for investigation")

    return obj


def overload(source_dict, target_dict):
    """
    Overloads elements of source_dict by elements of target_dict
    """
    # overload parameters
    for parameter in source_dict["parameters"]:
        if parameter != "server" and "." in source_dict["parameters"][parameter]:
            source_dict["parameters"][parameter] = deep_get(
                target_dict, source_dict["parameters"][parameter]
            )
    return source_dict


def overload_parameters(parameters_dict, target_dict):
    """
    Overloads elements of source_dict by elements of target_dict
    """
    # overload parameters
    for parameter in parameters_dict:
        if parameter != "server" and "." in parameters_dict[parameter]:
            parameters_dict[parameter] = deep_get(target_dict, parameters_dict[parameter])

    return parameters_dict


def overload_body(body, target_dict):
    """
    Overloads body by element of target_dict
    """
    # overload parameters
    if isinstance(body, str) and ".response.response" in body:
        return deep_get(target_dict, body)
    else:
        return body


def overload_body_parameters(body, parameters_dict):
    """
    Overloads body with parameters
    """
    for parameter in parameters_dict:
        if parameter[:4] == "body" and "." in parameter:
            temp_parameter = parameter[5:]
            try:
                body = deep_get_assign(body, temp_parameter, parameters_dict[parameter])
            except Exception:
                print(f"could not overload body ({parameter})")
    return body
