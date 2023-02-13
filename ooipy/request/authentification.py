"""
This module provides functions for the automatic authentication at the
OOI. The authentication is based on a username and a token, which will
be generatd after registering an account at
https://ooinet.oceanobservatories.org/.
"""

import os.path


def set_authentification(username, token):
    """
    Writes username and token to a text file. When requesting data from
    the OOI webservers, this textfile will automatically be accessed

    Parameters
    ----------
    username : str
        Username automatically generated by the OOI. It typically starts
        with "OOIAPI-...".
    token : str
        Token automatically generated by the OOI. It typically starts
        with "TEMP-TOKEN-...".
    """

    filename = "ooi_auth.txt"
    if not os.path.isfile(filename):
        file = open(filename, "w+")
        file.write("username\n" + username + "\n" + "token\n" + token)
        file.close()


def get_authentification():
    """
    Open ooi_auth.txt file and return the username and the token

    Returns
    -------
    (str, str)
    """

    filename = "ooi_auth.txt"
    if os.path.isfile(filename):
        file = open(filename)
        auth = file.readlines()
        username = auth[1].split("\n")[0]
        token = auth[3].split("\n")[0]
        file.close()
        return username, token
