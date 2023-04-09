# Code from ctxcore

# -*- coding: utf-8 -*-

import gzip
import os
import re
from collections.abc import Iterable
from collections.abc import Mapping as ABCMapping
from itertools import chain, repeat
from typing import FrozenSet, List, Mapping, Tuple

import attr
import yaml
from cytoolz import dissoc, first, keyfilter, memoize, merge, merge_with, second
from frozendict import frozendict


def convert(genes):
    # Genes supplied as dictionary.
    if isinstance(genes, ABCMapping):
        return frozendict(genes)
    # Genes supplied as iterable of (gene, weight) tuples.
    elif isinstance(genes, Iterable) and all(isinstance(n, tuple) for n in genes):
        return frozendict(genes)
    # Genes supplied as iterable of genes.
    elif isinstance(genes, Iterable) and all(isinstance(n, str) for n in genes):
        return frozendict(zip(genes, repeat(1.0)))


def openfile(filename: str, mode="r"):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode, encoding="utf-8")
    else:
        return open(filename, mode, encoding="utf-8")


@attr.s(frozen=True)
class GeneSignature(yaml.YAMLObject):
    """
    A class of gene signatures, i.e. a set of genes that are biologically related.
    """

    yaml_tag = "!GeneSignature"

    @classmethod
    def to_yaml(cls, dumper, data):
        dict_representation = {
            "name": data.name,
            "genes": list(data.genes),
            "weights": list(data.weights),
        }
        return dumper.represent_mapping(cls.yaml_tag, dict_representation, cls)

    @classmethod
    def from_yaml(cls, loader, node) -> "GeneSignature":
        data = loader.construct_mapping(node, cls)
        return GeneSignature(
            name=data["name"], gene2weight=zip(data["genes"], data["weights"])
        )

    @classmethod
    def from_gmt(
        cls, fname: str, field_separator: str = "\t", gene_separator: str = "\t"
    ) -> List["GeneSignature"]:
        """
        Load gene signatures from a GMT file.
        :param fname: The filename.
        :param field_separator: The separator that separates fields in a line.
        :param gene_separator: The separator that separates the genes.
        :return: A list of signatures.
        """
        # https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
        assert os.path.exists(fname), f'"{fname}" does not exist.'

        def signatures():
            with openfile(fname, "r") as file:
                for line in file:
                    if isinstance(line, (bytes, bytearray)):
                        line = line.decode()
                    if line.startswith("#") or not line.strip():
                        continue
                    name, _, genes_str = re.split(
                        field_separator, line.rstrip(), maxsplit=2
                    )
                    genes = genes_str.split(gene_separator)
                    yield GeneSignature(name=name, gene2weight=genes)

        return list(signatures())

    @classmethod
    def to_gmt(
        cls,
        fname: str,
        signatures: List["GeneSignature"],
        field_separator: str = "\t",
        gene_separator: str = "\t",
    ) -> None:
        """
        Save list of signatures as GMT file.
        :param fname: Name of the file to generate.
        :param signatures: The collection of signatures.
        :param field_separator: The separator that separates fields in a line.
        :param gene_separator: The separator that separates the genes.
        """
        # assert not os.path.exists(fname), "{} already exists.".format(fname)
        with openfile(fname, "wt") as file:
            for signature in signatures:
                genes = gene_separator.join(signature.genes)
                file.write(
                    f"{signature.name},{field_separator},{signature.metadata(',')},"
                    f"{field_separator}{genes}\n"
                )

    @classmethod
    def from_grp(cls, fname: str, name: str) -> "GeneSignature":
        """
        Load gene signature from GRP file.
        :param fname: The filename.
        :param name: The name of the resulting signature.
        :return: A signature.
        """
        # https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
        assert os.path.exists(fname), f'"{fname}" does not exist.'

        with openfile(fname, "r") as file:
            return GeneSignature(
                name=name,
                gene2weight=[
                    line.rstrip()
                    for line in file
                    if not line.startswith("#") and line.strip()
                ],
            )

    @classmethod
    def from_rnk(cls, fname: str, name: str, field_separator=",") -> "GeneSignature":
        """
        Reads in a signature from an RNK file. This format associates weights with the genes part of the signature.
        :param fname: The filename.
        :param name: The name of the resulting signature.
        :param field_separator: The separator that separates fields in a line.
        :return: A signature.
        """
        # https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
        assert os.path.exists(fname), f'"{fname}" does not exist.'

        def columns():
            with openfile(fname, "r") as file:
                for line in file:
                    if line.startswith("#") or not line.strip():
                        continue
                    columns = tuple(map(str.rstrip, re.split(field_separator, line)))
                    assert len(columns) == 2, "Invalid file format."
                    yield columns

        return GeneSignature(name=name, gene2weight=list(columns()))

    name: str = attr.ib()
    gene2weight: Mapping[str, float] = attr.ib(converter=convert)

    @name.validator
    def name_validator(self, attribute, value) -> None:
        if len(value) == 0:
            raise ValueError("A gene signature must have a non-empty name.")

    @gene2weight.validator
    def gene2weight_validator(self, attribute, value) -> None:
        if len(value) == 0:
            raise ValueError("A gene signature must have at least one gene.")

    @property
    @memoize
    def genes(self) -> Tuple[str, ...]:
        """
        Return genes in this signature. Genes are sorted in descending order according to weight.
        """
        return tuple(
            map(first, sorted(self.gene2weight.items(), key=second, reverse=True))
        )

    @property
    @memoize
    def weights(self) -> Tuple[float, ...]:
        """
        Return the weights of the genes in this signature. Genes are sorted in descending order according to weight.
        """
        return tuple(
            map(second, sorted(self.gene2weight.items(), key=second, reverse=True))
        )

    def metadata(self, field_separator: str = ",") -> str:
        """
        Textual representation of metadata for this signature.
        Is used as description when storing this signature as part of a GMT file.
        :param field_separator: the separator to use within fields.
        :return: The string representation of the metadata of this signature.
        """
        return ""

    def copy(self, **kwargs) -> "GeneSignature":
        # noinspection PyTypeChecker
        try:
            return GeneSignature(**merge(vars(self), kwargs))
        except TypeError:
            # Pickled gene signatures might still have nomenclature property.
            args = merge(vars(self), kwargs)
            del args["nomenclature"]
            return GeneSignature(**args)

    def rename(self, name: str) -> "GeneSignature":
        """
        Rename this signature.
        :param name: The new name.
        :return: the new :class:`GeneSignature` instance.
        """
        return self.copy(name=name)

    def add(self, gene_symbol: str, weight: float = 1.0) -> "GeneSignature":
        """
        Add an extra gene symbol to this signature.
        :param gene_symbol: The symbol of the gene.
        :param weight: The weight.
        :return: the new :class:`GeneSignature` instance.
        """
        return self.copy(
            gene2weight=list(chain(self.gene2weight.items(), [(gene_symbol, weight)]))
        )

    def union(self, other: "GeneSignature") -> "GeneSignature":
        """
        Creates a new :class:`GeneSignature` instance which is the union of this signature and the other supplied
        signature.
        The weight associated with the genes in the intersection is the maximum of the weights in the composing signatures.
        :param other: The other :class:`GeneSignature`.
        :return: the new :class:`GeneSignature` instance.
        """
        return self.copy(
            name=f"({self.name} | {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight=frozendict(
                merge_with(max, self.gene2weight, other.gene2weight)
            ),
        )

    def difference(self, other: "GeneSignature") -> "GeneSignature":
        """
        Creates a new :class:`GeneSignature` instance which is the difference of this signature and the supplied other
        signature.
        The weight associated with the genes in the difference are taken from this gene signature.
        :param other: The other :class:`GeneSignature`.
        :return: the new :class:`GeneSignature` instance.
        """
        return self.copy(
            name=f"({self.name} - {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight=frozendict(dissoc(dict(self.gene2weight), *other.genes)),
        )

    def intersection(self, other: "GeneSignature") -> "GeneSignature":
        """
        Creates a new :class:`GeneSignature` instance which is the intersection of this signature and the supplied other
        signature.
        The weight associated with the genes in the intersection is the maximum of the weights in the composing signatures.
        :param other: The other :class:`GeneSignature`.
        :return: the new :class:`GeneSignature` instance.
        """
        genes = set(self.gene2weight.keys()).intersection(set(other.gene2weight.keys()))
        return self.copy(
            name=f"({self.name} & {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight=frozendict(
                keyfilter(
                    lambda k: k in genes,
                    merge_with(max, self.gene2weight, other.gene2weight),
                )
            ),
        )

    def noweights(self):
        """
        Create a new gene signature with uniform weights, i.e. all weights are equal and set to 1.0.
        """
        return self.copy(gene2weight=self.genes)

    def head(self, n: int = 5) -> "GeneSignature":
        """
        Returns a gene signature with only the top n targets.
        """
        assert n >= 1, "n must be greater than or equal to one."
        genes = self.genes[
            0:n
        ]  # Genes are sorted in ascending order according to weight.
        return self.copy(gene2weight=keyfilter(lambda k: k in genes, self.gene2weight))

    def jaccard_index(self, other: "GeneSignature") -> float:
        """
        Calculate the symmetrical similarity metric between this and another signature.
        The JI is a value between 0.0 and 1.0.
        """
        ss = set(self.genes)
        so = set(other.genes)
        return float(len(ss.intersection(so))) / len(ss.union(so))

    def __len__(self) -> int:
        """
        The number of genes in this signature.
        """
        return len(self.genes)

    def __contains__(self, item: str) -> bool:
        """
        Checks if a gene is part of this signature.
        """
        return item in self.gene2weight.keys()

    def __getitem__(self, item: str) -> float:
        """
        Return the weight associated with a gene.
        """
        return self.gene2weight[item]

    def __str__(self) -> str:
        """
        Returns a readable string representation.
        """
        return f"[{','.join(self.genes)}]"

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation.
        """
        return '{}(name="{}",gene2weight=[{}])'.format(
            self.__class__.__name__,
            self.name,
            "["
            + ",".join(map(lambda g, w: f'("{g}",{w})', zip(self.genes, self.weights)))
            + "]",
        )