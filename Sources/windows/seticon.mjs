import * as ResEdit from "resedit";
import * as PELibrary from "pe-library";
import fs from "fs";
const exePath = process.argv[2];
const exe = PELibrary.NtExecutable.from(fs.readFileSync(exePath), { ignoreCert: true });
const res = PELibrary.NtExecutableResource.from(exe);
const ico = ResEdit.Data.IconFile.from(fs.readFileSync("icon.ico"));
const groups = ResEdit.Resource.IconGroupEntry.fromEntries(res.entries);
const id = groups.length ? groups[0].id : 1, lang = groups.length ? groups[0].lang : 1033;
ResEdit.Resource.IconGroupEntry.replaceIconsForResource(res.entries, id, lang, ico.icons.map(i => i.data));
const vi = ResEdit.Resource.VersionInfo.fromEntries(res.entries);
for (const v of vi) {
  v.setStringValues({ lang: 1033, codepage: 1200 }, {
    FileDescription: "Trait Fin", ProductName: "Trait Fin", CompanyName: "Trait Fin",
    OriginalFilename: "Trait Fin.exe", InternalName: "Trait Fin", LegalCopyright: ""
  });
  v.setFileVersion(1, 0, 0, 0, 1033); v.setProductVersion(1, 0, 0, 0, 1033);
  v.outputToResourceEntries(res.entries);
}
res.outputResource(exe);
fs.writeFileSync(exePath, Buffer.from(exe.generate()));
console.log("icon set", id, lang, vi.length);
