# Frontier-CS + FrontierSmith judge

`algorithmic/problems/` contains the merged problem bank: 180 numeric Frontier-CS
IDs and 200 `frontiersmith_*` IDs. The training recipe selects 190 Smith problems
after removing the ten published evaluation aliases, and evaluates on 179 FCS
problems (excluding problem 61). Serving the complete bank does not change those
training/evaluation splits.

On the judge machine, run:

```bash
cd Frontier-CS/algorithmic
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
export JUDGE_WORKERS=8 GJ_PARALLELISM=8
docker compose up -d --build
curl --fail --retry 10 --retry-connrefused --retry-delay 2 http://127.0.0.1:8081/problems
```

Docker privileged mode is required by go-judge. The service uses the upstream
container name `Competitive-Programming`; an existing container with that name
must be managed separately before deploying this one. The API has no authentication;
allow access only from a trusted network. Point all Harbor task verifiers to this
one service with `--frontier-judge-url http://<judge-host>:8081` during preparation.

Compose mounts local `problems/`, `data/` and `submissions/` directories. The judge
machine does not need the training filesystem, Python environment, models or Ray.
Use `docker compose logs -f` for logs and `docker compose down` to stop the judge.

## Sources

| Files | Upstream revision |
| --- | --- |
| Judge, Dockerfile, FCS problems, algorithmic reference and LICENSE | [Frontier-CS `59687e5`](https://github.com/FrontierCS/Frontier-CS/tree/59687e5428266a22a49cf51f981b912103bfa91d) |
| Smith problems, Harbor adapter and split manifests | [FrontierSmith `ac3046c`](https://github.com/FrontierCS/FrontierSmith/tree/ac3046cce7a270e3177cbb41376f36e03de1d518) |

The judge code is unchanged. The Dockerfile uses the npm bundled with Node.js 20
instead of upgrading to `npm@latest`, whose Node.js requirement breaks the upstream
build. Compose exposes the bind
address through `FRONTIERCS_JUDGE_BIND_IP` and concurrency through `JUDGE_WORKERS`
and `GJ_PARALLELISM`. A `.dockerignore` excludes the mounted problem bank and
runtime directories from the image build context. Original problem IDs,
statements, checkers and test data are preserved.
