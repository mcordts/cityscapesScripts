#!/usr/bin/env python3

from __future__ import print_function, absolute_import, division, unicode_literals

import appdirs
import argparse
import getpass
import json
import os
import requests
import shutil
import stat

from builtins import input


def login():
    appname = __name__.split('.')[0]
    appauthor = 'cityscapes'
    data_dir = appdirs.user_data_dir(appname, appauthor)
    credentials_file = os.path.join(data_dir, 'credentials.json')

    if os.path.isfile(credentials_file):
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
    else:
        username = input("Cityscapes username or email address: ")
        password = getpass.getpass("Cityscapes password: ")

        credentials = {
            'username': username,
            'password': password
        }

        store_question = "Store credentials unencrypted in '{}' [y/N]: "
        store_question = store_question.format(credentials_file)
        store = input(store_question).strip().lower()
        if store in ['y', 'yes']:
            os.makedirs(data_dir, exist_ok=True)
            with open(credentials_file, 'w') as f:
                json.dump(credentials, f)
            os.chmod(credentials_file, stat.S_IREAD | stat.S_IWRITE)

    session = requests.Session()
    r = session.get("https://www.cityscapes-dataset.com/login",
                    allow_redirects=False)
    r.raise_for_status()
    credentials['submit'] = 'Login'
    r = session.post("https://www.cityscapes-dataset.com/login",
                     data=credentials, allow_redirects=False)
    r.raise_for_status()

    # login was successful, if user is redirected
    if r.status_code != 302:
        if os.path.isfile(credentials_file):
            os.remove(credentials_file)
        raise Exception("Bad credentials.")

    return session


def get_available_packages(*, session):
    r = session.get(
        "https://www.cityscapes-dataset.com/downloads/?list", allow_redirects=False)
    r.raise_for_status()
    return r.json()


def list_available_packages(*, session):
    packages = get_available_packages(session=session)
    print("The following packages are available for download.")
    print("Please refer to https://www.cityscapes-dataset.com/downloads/ "
          "for additional packages and instructions on properly citing third party packages.")
    for p in packages:
        info = ' {} -> {}'.format(p['name'], p['size'])
        if p['thirdparty'] == '1':
            info += " (third party)"
        print(info)


def download_packages(*, session, package_names, destination_path):
    if not os.path.isdir(destination_path):
        raise Exception(
            "Destination path '{}' does not exist.".format(destination_path))

    packages = get_available_packages(session=session)
    name_to_id = {p['name']: p['packageID'] for p in packages}
    invalid_names = [n for n in package_names if n not in name_to_id]
    if invalid_names:
        raise Exception(
            "These packages do not exist or you don't have access: {}".format(invalid_names))

    for package_name in package_names:
        local_filename = os.path.join(destination_path, package_name)

        if os.path.exists(local_filename):
            raise Exception(
                "Destination file '{}' already exists.".format(local_filename))

        print("Downloading cityscapes package '{}' to '{}'".format(
            package_name, local_filename))
        package_id = name_to_id[package_name]
        url = "https://www.cityscapes-dataset.com/file-handling/?packageID={}".format(
            package_id)

        # download in chunks
        with session.get(url, allow_redirects=False, stream=True) as r:
            r.raise_for_status()
            assert r.status_code == 200

            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


def parse_arguments():
    description = "Download packages of the Cityscapes Dataset."
    epilog = "Requires an account that can be created via https://www.cityscapes-dataset.com/register/"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument('-l', '--list_available', action='store_true',
                        help="list available packages and exit")

    parser.add_argument('-d', '--destination_path', default='.',
                        help='destination path for downloads')

    parser.add_argument('package_name', nargs='*',
                        help="name of the packages to download")

    return parser.parse_args()


def main():
    args = parse_arguments()

    session = login()

    if args.list_available:
        list_available_packages(session=session)
        return

    download_packages(session=session, package_names=args.package_name,
                      destination_path=args.destination_path)


if __name__ == "__main__":
    main()
