"""Print the API's OpenAPI schema (used to generate the frontend's TypeScript types).

Run:  python -m api.export_openapi > ../plantvision-web/openapi.json
(the frontend's `npm run gen:api` does this for you)

The app is built without loading the model or opening the history database.
"""

import json

from api.main import create_app

if __name__ == "__main__":
    print(json.dumps(create_app(predictor=None, history=None).openapi(), indent=2, sort_keys=True))
