# Hosting the frontend on AWS

The frontend is a Create React App single-page application in `frontend/`.
The repository contains a root `amplify.yml` for AWS Amplify Hosting.

## Recommended deployment: Amplify Hosting

1. Push the branch containing `amplify.yml` to GitHub.
2. Open the [AWS Amplify console](https://console.aws.amazon.com/amplify/).
3. Choose **Create new app** → **Host web app**, connect GitHub, and select
   this repository and branch.
4. Mark the repository as a monorepo and set the app root to `frontend`.
5. Add this environment variable in Amplify:

   ```text
   REACT_APP_API_BASE_URL=https://<public-backend-domain>
   ```

   The value must be the HTTPS base URL of the deployed FastAPI backend. Do
   not add AWS access keys or secret keys; they would be embedded in the
   public JavaScript bundle.

6. Save and deploy. Amplify runs `npm ci`, `npm run build`, and publishes
   `frontend/build`.

After the first deployment, update the backend's `HCMAI_CORS_ORIGINS` with
the generated Amplify URL, then redeploy/restart the backend. For a custom
domain, use the final custom origin in that setting as well.

## SPA rewrite

The current UI uses one route, but keep this rewrite if client-side routes are
added later. In Amplify **Rewrites and redirects**, add:

```json
[
  {
    "source": "/<*>",
    "status": "200",
    "target": "/index.html",
    "condition": null
  }
]
```

## Current video-preview limitation

Search results and frame images use backend URLs. The inspector's video preview
currently attempts to create an S3 presigned URL in the browser. That requires
credentials and must not be enabled in a public Amplify build. Before using
video preview in production, expose a backend endpoint that issues a short-lived
signed URL after server-side authorization, and have the frontend call that
endpoint instead.
