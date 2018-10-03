"""
Usage:

from jwt_util import get_request_header

import os

xsuaa_base_url = os.environ['IT_XSUAA_BASE_URL']
client_id = os.environ['IT_CLIENT_ID']
client_secret = os.environ['IT_CLIENT_SECRET']

print(get_request_header(xsuaa_base_url, client_id, client_secret))

"""
import requests

def get_request_header(xsuaa_base_url, client_id, client_secret):
    response = requests.post(url=xsuaa_base_url + '/oauth/token',
                             data={'grant_type': 'client_credentials',
                                   'client_id': client_id,
                                   'client_secret': client_secret})
    access_token = response.json()["access_token"]
    return {'Authorization': 'Bearer {}'.format(access_token), 'Accept': 'application/json'}

