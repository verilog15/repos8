---
name: Share engine rules
about: Share your rules for search engines
title: 'My engine rules'
labels: 'Sites Rule'
assignees: ''

---

Paste rules here as:
<pre>
[
  {
    "type": "Image",
    "icon": "image",
    "sites": [
      {
        "name": "Google image",
        "url": "https://www.google.com/search?q=%s&tbm=isch",
        "match": "www\\.google\\..*tbm=isch"
      }
    ]
  }
]
</pre>
