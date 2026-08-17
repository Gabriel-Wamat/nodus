# Nodus

**Run models anywhere your cluster can.**

[![CI](https://github.com/Gabriel-Wamat/nodus/actions/workflows/ci.yml/badge.svg)](https://github.com/Gabriel-Wamat/nodus/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SDK Python para executar workloads de modelos em clusters SLURM usando OpenSSH, `rsync` e
venvs remotos. O núcleo não conhece clusters, modelos ou frameworks específicos: cada job
informa o comando, parâmetros, requisitos e arquivos necessários.

## Características do MVP

- API principal importável com `ClusterClient`.
- Configuração de conexão por variáveis de ambiente ou `~/.ssh/config`.
- Descoberta remota de nós, partições, GRES, features e módulos Python.
- Seleção da menor classe de GPU compatível ou preferência pela fila mais rápida.
- Venv remoto versionado por hash de `requirements` e módulo Python.
- Cache remoto de checkpoints por SHA-256: o mesmo peso não é reenviado.
- Snapshot imutável do projeto por hash de conteúdo.
- Estado persistente local em SQLite.
- Submissão, acompanhamento, logs, cancelamento e download.
- Nenhum container, daemon remoto ou senha armazenada pelo SDK.
- Nenhum passo exige `sudo`, instalação global, alteração do SLURM ou acesso administrativo.
- Bootstrap opcional com fingerprint e probes SLURM para clusters com GRES não tipado.

## Instalação

Durante o desenvolvimento:

```bash
python -m pip install -e .
```

Depois de publicar no GitHub:

```bash
python -m pip install \
  "nodus-runner @ git+ssh://git@github.com/Gabriel-Wamat/nodus.git@v0.1.0"
```

## Conexão

A opção preferida é deixar autenticação, proxy, porta e chave no OpenSSH. Os valores abaixo
são placeholders e devem ser substituídos pela configuração do cluster do usuário:

```sshconfig
Host my-cluster
    HostName login.cluster.example.org
    User SEU_USUARIO
    IdentityFile ~/.ssh/id_ed25519
```

```bash
export CLUSTER_SSH_ALIAS=my-cluster
export CLUSTER_REMOTE_ROOT='.cluster-model-runner'
export CLUSTER_INVENTORY_FILE="$PWD/profiles/example.inventory.json"
```

Alternativamente:

```bash
export CLUSTER_HOST=login.cluster.example.org
export CLUSTER_TRANSFER_HOST=transfer.cluster.example.org
export CLUSTER_USER=SEU_USUARIO
export CLUSTER_SSH_KEY=~/.ssh/id_ed25519
```

O SDK não lê senha de variável de ambiente. OpenSSH com chave/agent é mais seguro e funciona
de forma não interativa. Clusters com MFA podem ser usados por meio de uma conexão persistida
configurada pelo usuário no OpenSSH.

## Uso como SDK

```python
from nodus import ClusterClient, ResourceRequest

client = ClusterClient.from_env()

model = client.model(
    name="sam-inference",
    project=".",
    entrypoint="inference.py",
    requirements="requirements.lock",
    checkpoint="sam.safetensors",
    resources=ResourceRequest(
        min_vram_gb=20,
        gpu_count=1,
        cpus=8,
        ram_gb=32,
        time_limit="01:00:00",
    ),
)

job = model.submit(
    inputs={"image": "image.tif"},
    parameters={"prompt": "building", "precision": "fp16"},
)

print(job.id, job.slurm_id)
result = job.wait().download("results")
```

`wait()` reports state changes by default, including the SLURM pending reason, elapsed time,
selected node and exit code. Use `job.wait(progress=False)` for silent automation or pass
`on_update=` for structured callbacks.

Também é possível usar contratos explícitos `Project`, `Checkpoint` e `Venv`. `nodus` é o
namespace canônico; `cluster_model_runner` e a API anterior com `JobRequest` permanecem
compatíveis.

O script remoto usa o runtime tipado enviado automaticamente com cada job:

```python
from nodus.runtime import RuntimeRequest

request = RuntimeRequest.from_cli()
image = request.input("image")
checkpoint = request.checkpoint()
parameters = request.parameters

# execute a inferência...
request.write_result(data={"mask_count": 1}, artifacts=["mask.png"])
```

Isso é igual para PyTorch, Transformers, Diffusers, vLLM em batch e código próprio. O projeto
decide como carregar o modelo; o SDK administra cluster, arquivos e ciclo de vida.

## Cache do checkpoint

Na primeira chamada, o SDK calcula SHA-256 e publica o peso em:

```text
~/.cluster-model-runner/model_store/sha256/<HASH>/<ARQUIVO>
```

Uma segunda execução com o mesmo conteúdo apenas consulta `_READY` e reutiliza o caminho. Se
o conteúdo mudar, mesmo com o mesmo nome, surge uma nova entrada. A publicação usa diretório
temporário, lock remoto e rename para não expor upload incompleto.

## Venvs

Se `requirements=` for informado e `venv=` não for, `submit()` prepara/reutiliza o ambiente
antes de submeter o modelo. A identidade é derivada de:

```text
SHA-256(requirements + módulo Python)
```

Também é possível preparar antecipadamente:

```python
env = client.prepare_environment(
    "requirements.lock",
    name="vision",
    python_module="auto",
).wait()

job_request.venv = env.path
```

`auto` consulta `module -t avail Python` e usa a versão numericamente mais recente encontrada.
Se o sistema de módulos não estiver disponível no shell remoto, usa `python3` da partição de
instalação como base do venv. A partição pode vir de `CLUSTER_INSTALL_PARTITION`, de um perfil
com `installation: true` ou da descoberta dinâmica. Em todos os casos, o SDK valida a opção
com `sbatch --test-only`; o teste não cria um job e não requer privilégios administrativos.
Isso não garante sozinho a compatibilidade PyTorch/CUDA: as versões no lockfile precisam ser
compatíveis com o driver do cluster e devem ser validadas em um job GPU de smoke test.

## Seleção de GPU

O SDK sempre consulta o estado atual pelo `sinfo`. Quando o cluster anuncia GRES tipado, usa:

```text
--gres=gpu:<tipo>:<quantidade>
```

Quando o SLURM anuncia apenas `gpu:N`, um arquivo de inventário complementa modelo e VRAM. O
SDK solicita um nó, pede `gpu:N` e exclui classes incompatíveis. O scheduler continua decidindo
a colocação sem transformar a lista de candidatos em uma solicitação multi-nó.

O usuário escolhe explicitamente a necessidade do workload; esse valor nunca é substituído por
uma estimativa silenciosa:

```python
resources = ResourceRequest(min_vram_gb=24, policy="smallest-compatible")
```

Também existem `fastest-queue`, `safe` e `exact`. A política `exact` exige `gpu_type`. O valor
descoberto para cada GPU representa a capacidade segura depois da reserva configurável
`CLUSTER_GPU_VRAM_RESERVE_GB`; a capacidade física também permanece registrada no inventário.

O arquivo `profiles/example.inventory.json` documenta o formato agnóstico para complementar
VRAM, QoS e limites que o SLURM não exponha. Ele contém apenas nomes fictícios. Estado,
partições, CPU, RAM e disponibilidade continuam sendo descobertos em tempo de execução.

## Bootstrap dinâmico

`client.bootstrap()` calcula uma identidade estável do cluster, persiste o inventário em
`.nodus/clusters/<fingerprint>/inventory.json` e o reutiliza enquanto o TTL for válido. Primeiro
usa apenas `sinfo` e `scontrol`. Se faltar VRAM para a decisão solicitada, a política padrão
`when-needed` submete probes curtos com `nvidia-smi`; eles não instalam pacotes nem carregam
modelos.

```python
inventory = client.bootstrap()
inventory = client.bootstrap(probe_policy="representative", refresh=True)
inventory = client.bootstrap(probe_policy="all-nodes", refresh=True)
```

Nós equivalentes são agrupados por GRES, features, partições, CPU e RAM. O limite de
paralelismo, timeout e reserva segura de VRAM ficam centralizados nas variáveis `CLUSTER_*` do
`.env.example`. Probes pendentes reportam estado e motivo da fila no stderr.

## CLI auxiliar

```bash
nodus inspect
cluster-runner discover
nodus discover --full
nodus discover --refresh --show
nodus submit job.json --wait --download results
nodus env-create requirements.lock --name vision
nodus jobs
nodus status JOB_LOCAL_ID
nodus logs JOB_LOCAL_ID
nodus download JOB_LOCAL_ID --to results
nodus cancel JOB_LOCAL_ID
```

A CLI não possui lógica própria; ela chama o mesmo SDK importável.

## Exemplos reais

- [`examples/pytorch_vision`](examples/pytorch_vision): ResNet-18, checkpoint oficial,
  inferência CUDA e cache SHA-256 de arquivo.
- [`examples/transformers_llm`](examples/transformers_llm): `tiny-gpt2`, geração causal CUDA e
  cache SHA-256 de diretório multifile.

Os exemplos são cobertos por testes offline com transportes e scheduler simulados. Assets
grandes ficam em `.demo-cache` e são baixados pelos scripts de preparação; não entram no Git.

## Desenvolvimento

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
mypy src/cluster_model_runner src/nodus
pytest -q
```

Os testes usam transportes falsos e não exigem SSH nem acesso a cluster. As decisões de
arquitetura estão em [`docs/decisions`](docs/decisions), e limitações explícitas em
[`docs/limitations.md`](docs/limitations.md). A comprovação item a item está na
[`docs/evidence-matrix.md`](docs/evidence-matrix.md).

Contribuições e divulgação responsável de vulnerabilidades estão descritas em
[`CONTRIBUTING.md`](CONTRIBUTING.md) e [`SECURITY.md`](SECURITY.md).

## Limites atuais

- O MVP executa vLLM como processo batch. Serviço persistente não faz parte do contrato atual,
  pois pode depender de túneis e políticas específicas incompatíveis com a premissa sem admin.
- A criação automática de venv depende de uma partição autorizada com acesso ao índice de
  pacotes. O Nodus não tenta instalar Python ou bibliotecas globalmente.
- Dependências devem ser declaradas e fixadas pelo projeto. Não é seguro adivinhar versões de
  PyTorch, Transformers ou Diffusers sem conhecer o modelo e o driver remoto.
- Perfis de inventário são opcionais e pertencem à configuração do usuário; o núcleo não inclui
  nomes de nós, partições, QoS, modelos de GPU ou endpoints de um cluster específico.
