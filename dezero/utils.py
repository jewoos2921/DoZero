import os
import subprocess


# 이미지 변환까지 한번에
def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:]  # 확장자 (png, pdf 등)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


# 계산 그래프에서 DOT 언어로 변환하기
def _dot_var(v, verbose=False):
    # verbose=True 로 설정하면 ndarray 인스턴스의 '형상' 과 '타입' 도 함께 레이블로 출력
    dot_var = "{} [label='{}', color=orange, style=filled]\n"
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y는 약한 참조
    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # 계산 그래프에서 DOT 언어로 변환할 때는 '어떤 노드가 존재하는가' 와 '어떤 노드가 연결되는가' 가 문제이다.
            # 노드의 추적 순서는 중요하지 않다.
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g{\n" + txt + "}"
