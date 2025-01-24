from abc import ABC, abstractmethod

class Pessoa:
    def __init__(self, nome=''):
        self.nome = nome
    
class Funcionario(Pessoa, ABC):
    @abstractmethod
    def definirSalario(self, valor : float):
        pass
    @abstractmethod
    def definirCursoLecionado(self, nomeDoCurso : str):
        pass
    pass

class Estudante(Pessoa):
    @abstractmethod
    def definir(self):
        pass
    @abstractmethod
    def definirCursoFrequentado(self, nomeDoCurso : str):
        pass
    pass

class Aluno(Estudante):
    def __init__(self, nome='', matricula = 0, nomeDoCurso=''):
        __baseId = 0
        Pessoa.__init(self, nome)
        self.matricula= matricula
        self.nomeDoCurso = nomeDoCurso
        self.id = Aluno.__baseid
        Aluno.__baseId += 1
    
    def definirMatricula(self, numero : int):
        self.matricula = numero
    
    def definirCursoFrequentado(self, nomeDoCurso : str):
        self.nomeDoCurso = nomeDoCurso

class Professor(Funcionario):
    __baseId = 0
    def __init__(self, nome='', salario=0, nomeDoCurso=''):
        self.nome = nome
        self.salario = salario
        self.nomeDoCurso = nomeDoCurso
        self.id = Professor.__baseId    ##repetir para os outros
        Professor.__baseId += 1

    def definirSalario(self, valor:float):
        self.salario = valor

    def definirCursoSelecionado(self, nomeDoCurso, str):
        self.nomeDoCurso = nomeDoCurso

class AssistenteEnsino(Funcionario, Estudante):
    __baseId = 0
    def __init__(self, nome='', matricula =0, cursoFrequentado='', salario =0, cursoLecionado=''):
        Pessoa.__init__(self,nome)
        self.matricula = matricula
        self.cursoFrequentado = cursoFrequentado
        self.salario = salario
        self.cursoLecionado = cursoFrequentado
        self.id = AssistenteEnsino.__baseId
        AssistenteEnsino.__baseId += 1

    def definirSalario(self, valor : float):
        self.salario = valor
    def definirCursoLecionado(self, nomeDoCurso : str):
        self.cursoLecionado = nomeDoCurso
    def definirMatricula(self, numero : int):
        self.matricula = numero
    def definirCursoFrequentado(self, nomeDoCurso : str):
        self.cursoFrequentado = nomeDoCurso
    ##definir o diabo do str dps


if __name__ == '__main__':
    p = Pessoa('Joao')
    print(p.nome)

    prof = Professor('Jose', 1000, 'matematica')
    print(prof.id)
    print(prof.nome)
    prof2 = Professor('maria', 1000, 'fisica')
    print(prof2.id)
    print(prof2.nome)
    prof.definirSalario(2000)
    print(prof.salario)
    prof.definirCursoLecionado('telecom')
    print(prof.nomeDoCurso)
    
    a = Aluno('maria', 1234, 'telecom')
    b = Aluno('ana', 235, 'telecom')


