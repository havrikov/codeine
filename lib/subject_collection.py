#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, Any


class Subjects(object):
    """This class houses the available subject collection."""

    @property
    def all_subjects(self) -> Dict[str, Dict[str, Any]]:
        return {
            "javascript": self.javascript_subjects,
            "json-org": self.json_org_subjects,
            "url-w3c": self.uwl_w3c_subjects,
        }

    @property
    def uwl_w3c_subjects(self) -> Dict[str, Any]:
        return {
            "suffix": ".txt",
            "grammar": "url-w3c.scala",
            "drivers": {
                "autolink": "org.nibor.autolink",
                "jurl": "com.anthonynsimon.url",
                "url-detector": "com.linkedin.urls.detection",
            }
        }

    @property
    def json_org_subjects(self) -> Dict[str, Any]:
        return {
            "suffix": ".json",
            "grammar": "json-org.scala",
            "drivers": {
                "argo": "argo",
                "fastjson": "com.alibaba.fastjson",
                "genson": "com.owlike.genson",
                "gson": "com.google.gson",
                "json-flattener": "com.github.wnameless.json",
                "json-java": "org.json",
                "json-simple": "org.json.simple",
                "json-simple-cliftonlabs": "com.github.cliftonlabs.json_simple",
                "minimal-json": "com.eclipsesource.json",
                "pojo": "org.jsonschema2pojo",
            }
        }

    @property
    def javascript_subjects(self) -> Dict[str, Any]:
        return {
            "suffix": ".js",
            "grammar": "javascript.scala",
            "drivers": {
                # "closure": "com.google.javascript.jscomp",  # simply too slow :(
                "rhino": "org.mozilla.javascript",
            }
        }
