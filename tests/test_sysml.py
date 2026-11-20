from __future__ import annotations

import os
import tempfile
import unittest

from app.extract import extract
from app.relationships import extract_relationships
from app.sysml import from_model

XMI = """<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmi:version="2.1"
         xmlns:xmi="http://schema.omg.org/spec/XMI/2.1"
         xmlns:uml="http://www.omg.org/spec/UML/20110701"
         xmlns:sysml="http://www.omg.org/spec/SysML/20120322/SysML">
  <uml:Model xmi:id="_model" name="Air Vehicle">
    <packagedElement xmi:type="uml:Package" xmi:id="_struct" name="Structure">
      <packagedElement xmi:type="uml:Class" xmi:id="_veh" name="Vehicle"/>
      <packagedElement xmi:type="uml:Class" xmi:id="_prop" name="Propulsion">
        <ownedComment xmi:type="uml:Comment" xmi:id="_c1" body="Provides&#10;thrust."/>
        <generalization xmi:type="uml:Generalization" xmi:id="_gen" general="_veh"/>
        <ownedAttribute xmi:type="uml:Property" xmi:id="_thrust" name="maxThrust"
                        type="_real" aggregation="composite">
          <defaultValue xmi:type="uml:LiteralString" xmi:id="_dv" value="12000"/>
          <lowerValue xmi:type="uml:LiteralInteger" xmi:id="_lv" value="1"/>
          <upperValue xmi:type="uml:LiteralUnlimitedNatural" xmi:id="_uv" value="1"/>
        </ownedAttribute>
        <ownedAttribute xmi:type="uml:Property" xmi:id="_pump" name="pump"
                        aggregation="composite">
          <type xmi:idref="_pumpblk"/>
        </ownedAttribute>
      </packagedElement>
      <packagedElement xmi:type="uml:Class" xmi:id="_pumpblk" name="Fuel Pump"/>
      <packagedElement xmi:type="uml:DataType" xmi:id="_real" name="Real"/>
    </packagedElement>
    <packagedElement xmi:type="uml:Package" xmi:id="_reqs" name="Requirements">
      <packagedElement xmi:type="uml:Class" xmi:id="_req42" name="Thrust Margin"/>
    </packagedElement>
    <packagedElement xmi:type="uml:Abstraction" xmi:id="_sat">
      <client xmi:idref="_prop"/>
      <supplier xmi:idref="_req42"/>
    </packagedElement>
  </uml:Model>
  <sysml:Block xmi:id="_s1" base_Class="_prop"/>
  <sysml:Block xmi:id="_s2" base_Class="_veh"/>
  <sysml:Block xmi:id="_s5" base_Class="_pumpblk"/>
  <sysml:ValueType xmi:id="_s6" base_DataType="_real"/>
  <sysml:Requirement xmi:id="_s3" base_Class="_req42" Id="REQ-042"
                     Text="The vehicle shall hold a 10% thrust margin."/>
  <sysml:Satisfy xmi:id="_s4" base_Abstraction="_sat"/>
</xmi:XMI>
"""

BPMN = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  id="Defs_1" name="Order Handling">
  <bpmn:process id="Proc_1" name="Fulfilment">
    <bpmn:startEvent id="Start_1" name="Order received"/>
    <bpmn:task id="Task_1" name="Check stock"/>
    <bpmn:endEvent id="End_1" name="Order shipped"/>
    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start_1" targetRef="Task_1"/>
    <bpmn:sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="End_1"/>
  </bpmn:process>
</bpmn:definitions>
"""


def _write(directory: str, name: str, body: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return path


def _facts(text: str) -> set[tuple[str, str, str]]:
    return {(r.subject, r.predicate, r.object) for r in extract_relationships(text)}


class ModelExtractionTests(unittest.TestCase):
    def test_xmi_records_carry_hierarchy_stereotypes_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text = from_model(_write(directory, "vehicle.mdxml", XMI))

        assert text is not None
        self.assertIn("[model Air Vehicle]", text)
        self.assertIn("Qualified Name: Air Vehicle::Structure::Propulsion", text)
        self.assertIn("Type: Block", text)
        self.assertIn("Type: Requirement", text)
        self.assertIn("Id: REQ-042", text)
        self.assertIn("Text: The vehicle shall hold a 10% thrust margin.", text)
        self.assertIn("Parameter Name: maxThrust", text)
        self.assertIn("Multiplicity: 1..1", text)
        self.assertIn("Default: 12000", text)
        # A newline inside a value would end the record early.
        self.assertIn("Documentation: Provides thrust.", text)

    def test_xmi_references_become_typed_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text = from_model(_write(directory, "vehicle.mdxml", XMI))

        assert text is not None
        facts = _facts(text)
        expected = {
            ("Air Vehicle::Structure::Propulsion", "satisfies",
             "Air Vehicle::Requirements::Thrust Margin"),
            ("Air Vehicle::Structure::Propulsion", "specializes",
             "Air Vehicle::Structure::Vehicle"),
            ("Air Vehicle::Structure", "contains", "Air Vehicle::Structure::Propulsion"),
            ("Air Vehicle::Structure::Propulsion::maxThrust", "typed_by",
             "Air Vehicle::Structure::Real"),
            ("Air Vehicle::Structure::Propulsion", "has_part",
             "Air Vehicle::Structure::Fuel Pump"),
        }
        self.assertTrue(expected <= facts, expected - facts)
        # A value property is composition in SysML but is not a part.
        self.assertNotIn(
            ("Air Vehicle::Structure::Propulsion", "has_part", "Air Vehicle::Structure::Real"),
            facts,
        )

    def test_bpmn_sequence_flows_become_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text = from_model(_write(directory, "order.bpmn", BPMN))

        assert text is not None
        self.assertIn("Source Format: BPMN", text)
        facts = _facts(text)
        self.assertIn(
            ("Order Handling::Fulfilment::Order received", "flows_to",
             "Order Handling::Fulfilment::Check stock"),
            facts,
        )
        self.assertIn(
            ("Order Handling::Fulfilment::Check stock", "flows_to",
             "Order Handling::Fulfilment::Order shipped"),
            facts,
        )

    def test_extract_routes_model_extensions_and_sniffs_plain_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("vehicle.mdxml", "vehicle.xmi", "vehicle.uml", "vehicle.xml"):
                with self.subTest(name=name):
                    text = extract(_write(directory, name, XMI))
                    assert text is not None
                    self.assertIn("Relationship: satisfies", text)

            # An ordinary XML file still reads as its text, not as a model.
            plain = extract(_write(directory, "notes.xml", "<a><b>hello</b></a>"))
            self.assertEqual(plain, "hello")


if __name__ == "__main__":
    unittest.main()
