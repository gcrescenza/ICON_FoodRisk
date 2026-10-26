"""
ontology/owl_io.py
Scrittura e lettura di `food_ontology.owl` (OWL 2, serializzazione RDF/XML)
senza dipendenze esterne (solo libreria standard).

 - write_owl(path): genera il file a partire da ontology/spec.py.
   I 12 concetti BK sono classi definite (owl:equivalentClass) come
   intersezione di restrizioni su data property con datatype restriction
   (xsd:minInclusive / xsd:maxExclusive); i livelli low/high sono annotazioni
   (foodrisk:lowBelow, foodrisk:highFrom) sulle data property.
 - load_bk(path): rilegge dal FILE livelli e definizioni dei concetti.
 - summary(path): conta classi, proprieta', individui presenti nel file.

Il file prodotto e' RDF/XML standard; non e' stato aperto con Protege/owlready2
nell'ambiente di sviluppo (non disponibili): la coerenza e' verificata dal
lettore di questo modulo (round-trip spec -> file -> pipeline).
"""

import os
import xml.etree.ElementTree as ET

from ontology import spec

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
FR = spec.NS + "#"

for p, u in [("rdf", RDF), ("rdfs", RDFS), ("owl", OWL), ("xsd", XSD), ("foodrisk", FR)]:
    ET.register_namespace(p, u)


def _q(ns, tag): return f"{{{ns}}}{tag}"
def _res(parent, tag_ns, tag, iri):
    e = ET.SubElement(parent, _q(tag_ns, tag)); e.set(_q(RDF, "resource"), iri); return e
def _about(root, ns, tag, name):
    e = ET.SubElement(root, _q(ns, tag)); e.set(_q(RDF, "about"), FR + name); return e


def _restriction_value(feature, pred):
    """Predicato di livello -> (facet, valore) usando spec.LEVELS."""
    low, high = spec.LEVELS[feature]
    return {"low": ("maxExclusive", low), "notLow": ("minInclusive", low),
            "notHigh": ("maxExclusive", high), "high": ("minInclusive", high)}[pred]


def write_owl(path):
    root = ET.Element(_q(RDF, "RDF"))
    onto = ET.SubElement(root, _q(OWL, "Ontology")); onto.set(_q(RDF, "about"), spec.NS)

    for ap in ("lowBelow", "highFrom", "rawFeature"):
        _about(root, OWL, "AnnotationProperty", ap)

    for name, parent in spec.CLASS_TREE:
        c = _about(root, OWL, "Class", name)
        if parent: _res(c, RDFS, "subClassOf", FR + parent)

    for name, dom, rng in spec.OBJECT_PROPS:
        p = _about(root, OWL, "ObjectProperty", name)
        _res(p, RDFS, "domain", FR + dom); _res(p, RDFS, "range", FR + rng)

    for feat, (pname, xsdt) in spec.FEATURE_PROPS.items():
        p = _about(root, OWL, "DatatypeProperty", pname)
        _res(p, RDF, "type", OWL + "FunctionalProperty")
        _res(p, RDFS, "domain", FR + "Meal"); _res(p, RDFS, "range", XSD + xsdt)
        ET.SubElement(p, _q(FR, "rawFeature")).text = feat
        if feat in spec.LEVELS:
            low, high = spec.LEVELS[feat]
            for tag, v in (("lowBelow", low), ("highFrom", high)):
                e = ET.SubElement(p, _q(FR, tag)); e.set(_q(RDF, "datatype"), XSD + xsdt); e.text = str(v)

    for cname, cons in spec.BK_CONCEPTS.items():
        c = _about(root, OWL, "Class", cname)
        _res(c, RDFS, "subClassOf", FR + "Meal")
        eq = ET.SubElement(c, _q(OWL, "equivalentClass"))
        cls = ET.SubElement(eq, _q(OWL, "Class"))
        inter = ET.SubElement(cls, _q(OWL, "intersectionOf")); inter.set(_q(RDF, "parseType"), "Collection")
        m = ET.SubElement(inter, _q(OWL, "Class")); m.set(_q(RDF, "about"), FR + "Meal")
        for feat, pred in cons:
            pname, xsdt = spec.FEATURE_PROPS[feat]
            facet, val = _restriction_value(feat, pred)
            r = ET.SubElement(inter, _q(OWL, "Restriction"))
            _res(r, OWL, "onProperty", FR + pname)
            some = ET.SubElement(r, _q(OWL, "someValuesFrom"))
            dt = ET.SubElement(some, _q(RDFS, "Datatype"))
            _res(dt, OWL, "onDatatype", XSD + xsdt)
            wr = ET.SubElement(dt, _q(OWL, "withRestrictions")); wr.set(_q(RDF, "parseType"), "Collection")
            d = ET.SubElement(wr, _q(RDF, "Description"))
            f = ET.SubElement(d, _q(XSD, facet)); f.set(_q(RDF, "datatype"), XSD + xsdt); f.text = str(val)

    for name, cls, props in spec.INDIVIDUALS:
        i = _about(root, OWL, "NamedIndividual", name)
        _res(i, RDF, "type", FR + cls)
        for prop, targets in props.items():
            for t in targets: _res(i, FR, prop, FR + t)

    tree = ET.ElementTree(root); ET.indent(tree, space="  ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def ensure_ontology(path):
    if not os.path.exists(path):
        write_owl(path)
    return path


def _local(iri): return iri.replace("}", "#").split("#")[-1]


def load_bk(path):
    """
    Legge dal file OWL:
      levels   : {feature: (lowBelow, highFrom)}
      concepts : {nome_concetto: [(feature, op, valore, datatype), ...]}  (ordine di file)
      dtypes   : {feature: 'float'|'integer'}
    """
    root = ET.parse(path).getroot()
    prop_feat, dtypes, levels = {}, {}, {}
    for p in root.findall(_q(OWL, "DatatypeProperty")):
        name = _local(p.get(_q(RDF, "about")))
        feat = p.find(_q(FR, "rawFeature")).text
        prop_feat[name] = feat
        dtypes[feat] = _local(p.find(_q(RDFS, "range")).get(_q(RDF, "resource")))
        lo, hi = p.find(_q(FR, "lowBelow")), p.find(_q(FR, "highFrom"))
        if lo is not None and hi is not None:
            conv = int if dtypes[feat] == "integer" else float
            levels[feat] = (conv(lo.text), conv(hi.text))

    ops = {"maxExclusive": "<", "maxInclusive": "<=", "minInclusive": ">=", "minExclusive": ">"}
    concepts = {}
    for c in root.findall(_q(OWL, "Class")):
        inter = c.find(f"{_q(OWL,'equivalentClass')}/{_q(OWL,'Class')}/{_q(OWL,'intersectionOf')}")
        if inter is None: continue
        cons = []
        for r in inter.findall(_q(OWL, "Restriction")):
            feat = prop_feat[_local(r.find(_q(OWL, "onProperty")).get(_q(RDF, "resource")))]
            dt = r.find(f"{_q(OWL,'someValuesFrom')}/{_q(RDFS,'Datatype')}")
            d = dt.find(f"{_q(OWL,'withRestrictions')}/{_q(RDF,'Description')}")[0]
            conv = int if dtypes[feat] == "integer" else float
            cons.append((feat, ops[_local(d.tag)], conv(d.text), dtypes[feat]))
        concepts[_local(c.get(_q(RDF, "about")))] = cons
    return levels, concepts, dtypes


def summary(path):
    root = ET.parse(path).getroot()
    return {
        "classi": len(root.findall(_q(OWL, "Class"))),
        "object_property": len(root.findall(_q(OWL, "ObjectProperty"))),
        "data_property": len(root.findall(_q(OWL, "DatatypeProperty"))),
        "individui": len(root.findall(_q(OWL, "NamedIndividual"))),
        "concetti_bk": sum(1 for c in root.findall(_q(OWL, "Class"))
                           if c.find(_q(OWL, "equivalentClass")) is not None),
    }
