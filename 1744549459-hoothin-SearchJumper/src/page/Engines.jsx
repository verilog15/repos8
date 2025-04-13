import React from 'react';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Avatar from '@mui/material/Avatar';
import EditIcon from '@mui/icons-material/Edit';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import DeleteIcon from '@mui/icons-material/Delete';
import IconButton from '@mui/material/IconButton';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert from '@mui/material/Alert';
import Autocomplete from '@mui/material/Autocomplete';
// import SelectAllIcon from '@mui/icons-material/SelectAll';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import Checkbox from '@mui/material/Checkbox';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import InputBase from '@mui/material/InputBase';
import SearchIcon from '@mui/icons-material/Search';
import InputAdornment from '@mui/material/InputAdornment';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import DomainVerificationIcon from '@mui/icons-material/DomainVerification';


function saveConfigToScript (notification) {
    window.searchData.lastModified = new Date().getTime();
    var saveMessage = new CustomEvent('saveConfig', {
        detail: {
            searchData: window.searchData, 
            notification: !!notification
        }
    });
    document.dispatchEvent(saveMessage);
    window.saveToWebdav();
}

function TypeEdit(props) {
    const [typeData, setTypeData] = React.useState(props.data);

    React.useEffect(() => {
        setTypeData(props.data);
    }, [props.data])

    function closeTypeEdit(update, forceData) {
        if (update) {
            let targetData = forceData || typeData;
            if (!targetData.type) return props.handleAlertOpen(window.i18n('errorNoType'));
            for (let i = 0; i < window.searchData.sitesConfig.length; i++) {
                let type = window.searchData.sitesConfig[i];
                if (type.type === props.data.type) continue;
                if (type.type === targetData.type) {
                    return props.handleAlertOpen(window.i18n('errorSameType'));
                }
            }
            props.changeType(targetData);
        }
        props.closeHandler();
    }

    return (
        <Dialog open={props.typeOpen} onClose={() => {closeTypeEdit(false)}}>
            <DialogTitle
              sx={{display: 'flex', alignItems: 'center'}}
            >
              {window.i18n(typeData.type === '' ? 'addType' : 'editType')}
              {typeData.match === '0' ?
                  <Button
                    sx={{padding: '0 10px', cursor: 'pointer'}}
                    title={window.i18n('showIcon')}
                    onClick={e => {
                        closeTypeEdit(true, { ...typeData, match:'' });
                    }}
                  >
                    <VisibilityIcon/>
                  </Button>
                  :
                  <Button
                    sx={{padding: '0 10px', cursor: 'pointer'}}
                    title={window.i18n('hideIcon')}
                    onClick={e => {
                        closeTypeEdit(true, { ...typeData, match:'0' });
                    }}
                  >
                    <VisibilityOffIcon/>
                  </Button>
              }
            </DialogTitle>

            
            <DialogContent>
                <TextField
                    autoFocus
                    margin="dense"
                    id="name"
                    label={window.i18n('typeName')}
                    type="text"
                    fullWidth
                    variant="standard"
                    value={typeData.type}
                    onChange={e => {
                        setTypeData({ ...typeData, type:e.target.value });
                    }}
                />
                <TextField
                    margin="dense"
                    id="icon"
                    label={window.i18n('typeIcon')}
                    type="text"
                    fullWidth
                    variant="standard"
                    value={typeData.icon}
                    onChange={e => {
                        setTypeData({ ...typeData, icon:e.target.value });
                    }}
                    InputProps={{
                      endAdornment: (
                          <InputAdornment position="end">
                            <input
                                accept="image/*"
                                style={{ display: "none" }}
                                id="upload-type-icon"
                                type="file"
                                onChange={event => {
                                    let file = event.target.files && event.target.files[0];
                                    if (!file) return;
                                    if (file.size > 51200 && !window.confirm(window.i18n('imgTooBig'))) {
                                        event.target.value = "";
                                        return;
                                    }
                                    let reader = new FileReader();
                                    reader.readAsDataURL(file);
                                    reader.onload = function() {
                                        setTypeData({ ...typeData, icon:reader.result });
                                    };
                                }}
                            />
                            <label htmlFor="upload-type-icon">
                                <IconButton
                                  edge="end"
                                  component="span"
                                >
                                    <FileUploadIcon/>
                                </IconButton>
                            </label>
                          </InputAdornment>
                        ),
                      inputProps: { spellCheck: 'false' }
                    }}
                    
                />
                <DialogContentText>
                    {window.i18n('iconTips')}
                </DialogContentText>
                <Accordion sx={{margin: '0 -16px!important'}}>
                    <AccordionSummary
                      sx={{background: '#d1d1d120', minHeight: '45px!important', maxHeight: '45px!important'}}
                      expandIcon={<ExpandMoreIcon />}>
                      <Typography align="center" sx={{width: '100%'}}>{window.i18n('moreOptions')}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <TextField
                            margin="dense"
                            id="description"
                            label={window.i18n('description')}
                            type="text"
                            fullWidth
                            variant="standard"
                            value={typeData.description}
                            onChange={e => {
                                setTypeData({ ...typeData, description:e.target.value });
                            }}
                        />
                        <TextField
                            margin="dense"
                            id="match"
                            label={window.i18n('typeMatch')}
                            type="text"
                            fullWidth
                            variant="standard"
                            placeholder="www\.google\.com"
                            value={typeData.match}
                            onChange={e => {
                                setTypeData({ ...typeData, match:e.target.value });
                            }}
                            inputProps={{ spellCheck: 'false' }}
                        />
                        <DialogContentText>
                            {window.i18n('typeMatchTips')}
                        </DialogContentText>
                        <Box style={{textAlign: "center", border: "1px solid rgba(0, 0, 0, 0.42)", borderRadius: "10px"}}>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectTxt} 
                                            name="enableSelTxt"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectTxt:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelTxt')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectImg} 
                                            name="enableSelImg"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectImg:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelImg')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectVideo} 
                                            name="enableSelVideo"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectVideo:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelVideo')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectAudio} 
                                            name="enableSelAudio"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectAudio:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelAudio')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectLink} 
                                            name="enableSelLink"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectLink:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelLink')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                            <FormControl sx={{ m: 1, minWidth: 80 }}>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={typeData.selectPage} 
                                            name="enableSelPage"
                                            onClick={e => {
                                                setTypeData({ ...typeData, selectPage:e.target.checked });
                                            }}
                                        />
                                    }
                                    label={window.i18n('typeEnableSelPage')}
                                    labelPlacement="top"
                                />
                            </FormControl>
                        </Box>
                        <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'nowrap', mb: 1}}>
                            <TextField
                                sx={{ minWidth: 100, maxWidth: 150 }}
                                margin="dense"
                                id="match"
                                label={window.i18n('typeShotcut')}
                                type="text"
                                variant="outlined"
                                value={(typeData.shortcut || "").replace(/Key|Digit/, "").replace(/Backquote/, "`").replace(/Minus/, "-").replace(/Equal/, "=").replace(/ArrowUp/, "↑").replace(/ArrowDown/, "↓").replace(/ArrowLeft/, "←").replace(/ArrowRight/, "→")}
                                inputProps={{ readOnly: 'readonly' }}
                                onKeyDown={e => {
                                    if (/^(Control|Alt|Meta|Shift)/.test(e.key)) {
                                        return;
                                    }
                                    setTypeData({
                                        ...typeData,
                                        ctrl: e.ctrlKey,
                                        alt: e.altKey,
                                        shift: e.shiftKey,
                                        meta: e.metaKey,
                                        shortcut: (e.key === 'Escape' || e.key === 'Backspace') ? '' : (e.code || e.key)
                                    });
                                }}
                            />
                            <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'wrap'}}>
                                <FormControl sx={{ minWidth: 60 }}>
                                    <FormControlLabel className="keyboardBtn"
                                        control={
                                            <Switch 
                                                checked={typeData.ctrl} 
                                                name="ctrl"
                                                onClick={e => {
                                                    setTypeData({...typeData, ctrl: e.target.checked});
                                                }}
                                            />
                                        }
                                        label='Ctrl'
                                        labelPlacement="bottom"
                                    />
                                </FormControl>
                                <FormControl sx={{ minWidth: 60 }}>
                                    <FormControlLabel className="keyboardBtn"
                                        control={
                                            <Switch 
                                                checked={typeData.alt} 
                                                name="alt"
                                                onClick={e => {
                                                    setTypeData({...typeData, alt: e.target.checked});
                                                }}
                                            />
                                        }
                                        label='Alt'
                                        labelPlacement="bottom"
                                    />
                                </FormControl>
                                <FormControl sx={{ minWidth: 60 }}>
                                    <FormControlLabel className="keyboardBtn"
                                        control={
                                            <Switch 
                                                checked={typeData.shift} 
                                                name="shift"
                                                onClick={e => {
                                                    setTypeData({...typeData, shift: e.target.checked});
                                                }}
                                            />
                                        }
                                        label='Shift'
                                        labelPlacement="bottom"
                                    />
                                </FormControl>
                                <FormControl sx={{ minWidth: 60 }}>
                                    <FormControlLabel className="keyboardBtn"
                                        control={
                                            <Switch 
                                                checked={typeData.meta} 
                                                name="meta"
                                                onClick={e => {
                                                    setTypeData({...typeData, meta: e.target.checked});
                                                }}
                                            />
                                        }
                                        label='Meta'
                                        labelPlacement="bottom"
                                    />
                                </FormControl>
                            </Box>
                        </Box>
                        <FormControl sx={{ minWidth: 80 }}>
                            <InputLabel>{window.i18n('openSelect')}</InputLabel>
                            <Select
                                value={typeData.openInNewTab}
                                name="openInNewTab"
                                onChange={(event: SelectChangeEvent) => {
                                    setTypeData({ ...typeData, openInNewTab:event.target.value });
                                }}
                                autoWidth
                                label={window.i18n('openSelect')}
                            >
                                <MenuItem value={-1}>{window.i18n("openInDefaultOption")}</MenuItem>
                                <MenuItem value={1}>{window.i18n("openInNewTabOption")}</MenuItem>
                                <MenuItem value={0}>{window.i18n("openInCurrentOption")}</MenuItem>
                                <MenuItem value={4}>{window.i18n("openInBackOption")}</MenuItem>
                                <MenuItem value={2}>{window.i18n("openInIncognitoOption")}</MenuItem>
                                <MenuItem value={3}>{window.i18n("openInMinWindowOption")}</MenuItem>
                            </Select>
                        </FormControl>
                    </AccordionDetails>
                </Accordion>
            </DialogContent>
            <DialogActions>
                <Button variant="outlined" color="error" startIcon={<DeleteIcon />} onClick={props.handleDeleteType}>{window.i18n('delete')}</Button>
                <Button onClick={() => { closeTypeEdit(false) }}>{window.i18n('cancel')}</Button>
                <Button onClick={() => { closeTypeEdit(true) }}>{window.i18n(typeData.type === '' ? 'add' : 'save')}</Button>
            </DialogActions>
        </Dialog>
    );
}

var dragTargetLine;
function hideDragLine() {
    if (!dragTargetLine) dragTargetLine = document.querySelector(`#dragTargetLine`);
    if (dragTargetLine) {
        dragTargetLine.style.display = "";
    }
}

class ChildSiteIcons extends React.Component {
    shouldComponentUpdate(nextProps, nextState){
        return nextProps.sites !== this.props.sites || nextProps.checkeds.length !== this.checkeds.length || !nextProps.checkeds.every((value, index) => value === this.checkeds[index]);
    }

    getIcon(site){
        let icon = "";
        let isClone = site.url.indexOf('[') === 0;
        if (isClone) {
            try {
                let siteNames = JSON.parse(site.url);
                if (siteNames.length === 1) {
                    for (let i = 0; i < window.searchData.sitesConfig.length; i++) {
                        let typeData = window.searchData.sitesConfig[i];
                        let sites = typeData.sites;
                        for (let j = 0; j < sites.length; j++) {
                            let _site = sites[j];
                            if (_site.url.indexOf('[') !== 0 && _site.name === siteNames[0]) {
                                icon = _site.icon || _site.url.replace(new RegExp('^(showTips:)?(https?://[^/]*/)[\\s\\S]*$'), "$2favicon.ico");
                                return (/^http/.test(icon) && window.cacheIcon[icon]) || icon;
                            }
                        }
                    }
                }
            } catch(e) {
                console.log(e);
            }
        }
        
        if (site.icon) {
            icon = site.icon;
        } else if (/^(showTips:)?https?:/.test(site.url)) {
            icon = site.url.replace(new RegExp('^(showTips:)?(https?://[^/]*/)[\\s\\S]*$'), "$2favicon.ico");
        }
        return (/^http/.test(icon) && window.cacheIcon[icon]) || icon;
    }

    dragOver(e) {
        e.preventDefault();
        if (!dragTargetLine) dragTargetLine = document.querySelector(`#dragTargetLine`);
        if (dragTargetLine) {
            dragTargetLine.style.display = "block";
            let target = e.currentTarget;
            target.parentNode.parentNode.appendChild(dragTargetLine);
            let isRight = e.clientX > getOffsetLeft(target) + target.offsetWidth / 2;
            dragTargetLine.style.top = target.offsetTop + "px";
            dragTargetLine.style.left = (isRight ? target.offsetLeft + target.offsetWidth : target.offsetLeft) + "px";
        }
    }

    getSliceText(text, maxLen = 10) {
        if (!text) return text;
        let result = "", len = 0;
        text = Array.from(text);
        for (let i = 0; i < text.length; i++) {
            let curChar = text[i];
            result += curChar;
            let charCode = curChar.charCodeAt(0);
            if (charCode >= 0 && charCode <= 128) {
                len++;
            } else {
                len = len + 2;
            }
            if (len >= maxLen) break;
        }
        return result;
    }

    render() {
        this.checkeds = [...this.props.checkeds];
        return (
            <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'wrap'}}>
                {this.props.sites.map((site, i) => (
                    <SiteIcon {...this.props} checked={this.props.checkeds[i]} site={site} key={i} i={i} getIcon={this.getIcon} getSliceText={this.getSliceText} dragOver={this.dragOver} />
                    // <Box className="site-icon" key={i}>
                    //     <Checkbox
                    //         onChange={e => {
                    //             this.props.checkChange(e, i);
                    //         }}
                    //         checked={this.props.checkeds[i]}
                    //     />
                    //     <IconButton className={(site.match === '0' ? 'hideIcon' : '')} sx={{fontSize: '1rem', flexDirection: 'column'}} draggable='true' onDragLeave={e => {hideDragLine()}} onDrop={e => {hideDragLine();this.props.changeSitePos(site, e);}} onDragStart={e => {e.dataTransfer.setData("data", JSON.stringify(site));}} onDragOver={e => this.dragOver(e)} key={site.name} title={site.description || site.name}  onClick={() => { this.props.openSiteEdit(site) }}>
                    //         <Avatar sx={{m:1}} alt={site.name} src={(!this.props.tooLong && this.getIcon(site)) || ''} >{this.props.tooLong || !site.name ? '🌐' : (/^[\s\w]{2}/.test(site.name) ? site.name.slice(0, 2) : Array.from(site.name)[0])}</Avatar>{this.getSliceText(site.name)}
                    //     </IconButton>
                    // </Box>
                ))}
                <div id="dragTargetLine"/>
                <IconButton color="primary" key='addType' onClick={() => { this.props.openSiteEdit(false); }}>
                    <AddCircleOutlineIcon sx={{fontSize: '50px'}} />
                </IconButton>
            </Box>
        );
    }
}

const SiteIcon = React.memo(function SiteIcon(props) {
    return (
        <Box className="site-icon" key={props.i}>
            <Checkbox
                onChange={e => {
                    props.checkChange(e, props.i);
                }}
                checked={props.checked}
            />
            <IconButton className={(props.site.match === '0' ? 'hideIcon' : '')} sx={{fontSize: '1rem', flexDirection: 'column'}} draggable='true' onDragLeave={e => {hideDragLine()}} onDrop={e => {hideDragLine();props.changeSitePos(props.site, e);}} onDragStart={e => {e.dataTransfer.setData("data", JSON.stringify(props.site));}} onDragOver={e => props.dragOver(e)} key={props.site.name} title={props.site.description || props.site.name}  onClick={() => { props.openSiteEdit(props.site) }}>
                <Avatar sx={{m:1}} alt={props.site.name} src={(!props.tooLong && props.getIcon(props.site)) || ''} >{props.tooLong || !props.site.name ? '🌐' : (/^[\s\w]{2}/.test(props.site.name) ? props.site.name.slice(0, 2) : Array.from(props.site.name)[0])}</Avatar>{props.getSliceText(props.site.name)}
            </IconButton>
        </Box>
    );
}, (oldProps, newProps) => {
  return (
    oldProps.site === newProps.site &&
    oldProps.i === newProps.i &&
    oldProps.checked === newProps.checked
  );
});
// function ChildSiteIcon({sites, changeSitePos, tooLong, checkChange, checked, openSiteEdit}) {
//     console.log("sitesIconRender");
//     return (
//         <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'wrap'}}>
//             {sites.map((site, i) => (
//                 <Box className="site-icon" key={i}>
//                     <Checkbox
//                         onChange={e => checkChange(e, i)}
//                         checked={checked[i]}
//                     />
//                     <IconButton sx={{fontSize: '1rem', flexDirection: 'column'}} draggable='true' onDrop={e => {changeSitePos(site, this.dragSite)}} onDragStart={e => {this.dragSite = site}} onDragOver={e => {e.preventDefault()}} key={site.name} title={site.name}  onClick={() => { openSiteEdit(site) }}>
//                         {tooLong ? '' : <Avatar sx={{m:1}} alt={site.name} src={(!tooLong && (site.icon || (/^http/.test(site.url) && site.url.replace(new RegExp('(https?://[^/]*/).*$'), "$1favicon.ico"))))||''} />}{site.name.length > 10 ? site.name.slice(0, 10) : site.name}
//                     </IconButton>
//                 </Box>
//             ))}
//             <IconButton color="primary" key='addType' onClick={() => { openSiteEdit(false); }}>
//                 <AddCircleOutlineIcon sx={{fontSize: '50px'}} />
//             </IconButton>
//         </Box>
//     );
// }

// const MemoSiteIcon = React.memo(ChildSiteIcon);

class SitesList extends React.Component {
    constructor(props) {
        super(props);
        this.state = {data: props.data, isOpenSiteEdit: false, isOpenLocalApp:false, currentSite: siteObject(false), checkeds: Array(props.data.sites.length).fill(false), cloneSite: false};

        this.editSite = null;
        this.openTypeEdit = props.openTypeEdit;
        this.index = props.index;
        this.handleAlertOpen = props.handleAlertOpen;

        this.openSiteEdit = this.openSiteEdit.bind(this);
        this.closeSiteEdit = this.closeSiteEdit.bind(this);
        this.handleDeleteSite = this.handleDeleteSite.bind(this);
        this.changeSitePos = this.changeSitePos.bind(this);
        this.tooLong = props.data.sites && props.data.sites.length > 100;
        this.batchSelect = false;
        var downloadEle = document.createElement('a');
        downloadEle.target = "_blank";
        this.downloadEle = downloadEle;
    }

    getMinSiteData(siteData) {
        let obj = {};
        if (siteData.name) {
            obj.name = siteData.name;
        }
        if (siteData.url) {
            obj.url = siteData.url;
        }
        if (siteData.icon) {
            obj.icon = siteData.icon;
        }
        if (siteData.keywords) {
            obj.keywords = siteData.keywords;
        }
        if (siteData.kwFilter) {
            obj.kwFilter = siteData.kwFilter;
        }
        if (siteData.description) {
            obj.description = siteData.description;
        }
        if (siteData.match) {
            obj.match = siteData.match;
        }
        if (siteData.charset) {
            obj.charset = siteData.charset;
        }
        if (siteData.shortcut) {
            obj.shortcut = siteData.shortcut;
        }
        if (siteData.ctrl) {
            obj.ctrl = siteData.ctrl;
        }
        if (siteData.alt) {
            obj.alt = siteData.alt;
        }
        if (siteData.shift) {
            obj.shift = siteData.shift;
        }
        if (siteData.meta) {
            obj.meta = siteData.meta;
        }
        if (siteData.nobatch) {
            obj.nobatch = siteData.nobatch;
        }
        if (siteData.hideNotMatch) {
            obj.hideNotMatch = siteData.hideNotMatch;
        }
        if (siteData.openInNewTab !== -1) {
            obj.openInNewTab = siteData.openInNewTab;
        }
        return obj;
    }

    openSiteEdit(data) {
        this.editSite = data;
        let currentSite = siteObject(data);
        this.setState(prevState => ({
            isOpenSiteEdit: true,
            currentSite: currentSite
        }));
    }

    closeSiteEdit(update) {
        if (update) {
            let currentType = window.searchData.sitesConfig[this.index];
            if (!this.state.currentSite.name) return this.handleAlertOpen(window.i18n('needName'));
            if (!this.state.currentSite.url) return this.handleAlertOpen(window.i18n('needUrl'));
            if (this.state.currentSite.icon && !/^(https?|ftp|data):|^0$/.test(this.state.currentSite.icon)) return this.handleAlertOpen(window.i18n('wrongImg'));
            if (/^\[/.test(this.state.currentSite.url)) {
                try {
                    JSON.parse(this.state.currentSite.url);
                } catch (e) {
                    return this.handleAlertOpen(e.toString());
                }
            }
            let isClone = this.state.currentSite.url.indexOf('[') === 0;
            for (let i = 0; i < window.searchData.sitesConfig.length; i++) {
                let typeData = window.searchData.sitesConfig[i];
                let sites = typeData.sites;
                for (let j = 0; j < sites.length; j++) {
                    let site = sites[j];
                    if (site.url === this.editSite.url) continue;
                    if (!isClone && site.url === this.state.currentSite.url) {
                        return this.handleAlertOpen(window.i18n('sameSiteUrl'));
                    }
                    if (this.state.currentSite.shortcut) {
                        if (site.shortcut === this.state.currentSite.shortcut) {
                            return this.handleAlertOpen(window.i18n('sameShortcut', site.name));
                        }
                    }
                    if (!isClone && site.url.indexOf('[') !== 0 && site.name === this.state.currentSite.name) {
                        return this.handleAlertOpen(window.i18n('sameSiteName', typeData.type));
                    }
                }
            }
            if (this.editSite) {
                let newSites = this.state.data.sites.map(site => {
                    if (site.url === this.editSite.url) {
                        return this.getMinSiteData(this.state.currentSite);
                    }
                    return site;
                })
                let newType = {...currentType, sites: newSites};
                let changeName = this.editSite.name !== this.state.currentSite.name && !/^\[/.test(this.editSite.url);
                window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                    let returnData = data;
                    if (currentType.type === data.type) {
                        returnData = {...data, sites: newSites};
                    }
                    if (changeName) {
                        returnData.sites = returnData.sites.map(site => {
                            if (/^\[/.test(site.url)) {
                                site.url = site.url.replaceAll('"' + this.editSite.name + '"', '"' + this.state.currentSite.name + '"');
                            }
                            return site;
                        });
                    }
                    return returnData;
                });
                this.setState(prevState => ({
                    data: newType
                }));
            } else {
                let newSites = this.state.data.sites.concat([this.getMinSiteData(this.state.currentSite)])
                let newType = {...currentType, sites: newSites};
                window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                    if (currentType.type === data.type) {
                        return newType;
                    }
                    return data;
                });
                this.setState(prevState => ({
                    data: newType
                }));
            }
            saveConfigToScript();
        }
        this.setState(prevState => ({
            isOpenSiteEdit: false
        }));
    }

    handleDeleteSite() {
        this.setState(prevState => ({
            isOpenSiteEdit: false
        }));
        let currentType = window.searchData.sitesConfig[this.index];
        let newSites = this.state.data.sites.filter(site => {
            return (site.url !== this.editSite.url);
        });
        let newType = {...currentType, sites: newSites};
        window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
            let returnData = data;
            if (currentType.type === data.type) {
                returnData = newType;
            }
            if (!/^\[/.test(this.editSite.url)){
                returnData.sites = returnData.sites.map(site => {
                    if (/^\[/.test(site.url)) {
                        try {
                            let namesArr = JSON.parse(site.url);
                            namesArr = namesArr.filter(n => {
                                return (n !== this.editSite.name);
                            });
                            site.url = namesArr.length === 0 ? '' : JSON.stringify(namesArr);
                        } catch (e) {
                            console.log(e);
                        }
                    }
                    return site;
                });
            }
            returnData.sites = returnData.sites.filter(site => {
                return site.url !== '';
            });
            return returnData;
        });
        this.setState(prevState => ({
            data: newType
        }));
        saveConfigToScript();
    }

    changeSitePos(targetSite, event) {
        let dragSite;
        try {
            dragSite = JSON.parse(event.dataTransfer.getData("data"));
        } catch (e) {
            console.log(e);
            return;
        }
        let target = event.currentTarget;
        let isRight = event.clientX > getOffsetLeft(target) + target.offsetWidth / 2;
        if (!targetSite || !dragSite || !targetSite.url || !dragSite.url) return;
        if (targetSite.url === dragSite.url) return;
        let currentType = window.searchData.sitesConfig[this.index];
        let newSites = this.state.data.sites.filter(site => {
            return (site.url !== dragSite.url);
        })
        for (let i = 0; i < newSites.length; i++) {
            if (newSites[i].url === targetSite.url) {
                newSites.splice(parseInt(i) + (isRight ? 1 : 0), 0, dragSite);
                break;
            }
        }
        let newType = {...currentType, sites: newSites};
        window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
            if (currentType.type === data.type) {
                return newType;
            }
            return data;
        });
        this.setState(prevState => ({
            data: newType
        }));
        saveConfigToScript();
    }

    render() {
        return (
            <Box elevation={5} component={Paper} sx={{p: '20px', mt: 2}} className='site-list-box'>
                <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', minHeight: 50, flexWrap: 'wrap' }}>
                    <IconButton title={window.i18n('editType')} color="primary" key='editType' onClick={() => { this.openTypeEdit(this.index) }}>
                        <EditIcon />
                    </IconButton>
                    <IconButton title={window.i18n('batchAction')} color="primary" key='mulSelType' onClick={() => { 
                        this.batchSelect = !this.batchSelect; 
                        let siteListBox = document.querySelector(".site-list-box");
                        if (this.batchSelect) {
                            siteListBox.classList.add("batch-edit");
                        } else {
                            siteListBox.classList.remove("batch-edit");
                        }
                    }}>
                        <CheckCircleIcon />
                    </IconButton>
                    <Button onClick={() => { 
                        this.setState(prevState => ({ 
                            checkeds: Array(this.state.data.sites.length).fill(false)
                        }));
                    }}>{window.i18n('cancel')}</Button>
                    <Button onClick={() => { 
                        this.setState(prevState => ({ 
                            checkeds: Array(this.state.data.sites.length).fill(true)
                        }));
                    }}>{window.i18n('selectAll')}</Button>
                    <Button onClick={() => { 
                        this.setState(prevState => ({ 
                            checkeds: prevState.checkeds.map(v => !v)
                        }));
                    }}>{window.i18n('invert')}</Button>
                    <Button variant="outlined" color="error" sx={{ml: 'auto', height: 35}} startIcon={<DeleteIcon />} onClick={() => {
                        let newSites = this.state.data.sites.filter((site, i) => {
                            return (this.state.checkeds[i] !== true);
                        })
                        if (newSites.length === this.state.data.sites.length || !window.confirm(window.i18n('deleteConfirm'))) return;
                        let newType = {...window.searchData.sitesConfig[this.index], sites: newSites};
                        window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                            if (this.state.data.type === data.type) {
                                return newType;
                            }
                            return data;
                        });
                        this.setState(prevState => ({
                            data: newType,
                            isOpenSiteEdit: false,
                            checkeds: Array(newSites.length).fill(false)
                        }));
                        saveConfigToScript();
                    }}>{window.i18n('delete')}</Button>
                    <FormControl sx={{ ml: 1, mr: 0 }}>
                        <InputLabel>{window.i18n('moveTo')}</InputLabel>
                        <Select
                            value={0}
                            name="x"
                            sx={{height: 35}}
                            onChange={(event: SelectChangeEvent) => {
                                let moveSites = this.state.data.sites.filter((site, i) => {
                                    return (this.state.checkeds[i] === true);
                                })
                                if (moveSites.length === 0) return;
                                if (this.state.cloneSite) {
                                    if (!window.confirm(window.i18n('cloneConfirm', event.target.value))) return;
                                    let cloneToGroup = moveSites.length !== 1 && window.confirm(window.i18n('cloneAction'));
                                    let cloneSites;
                                    if (cloneToGroup) {
                                        let groupName = window.prompt(window.i18n('groupName'));
                                        let groupUrlArr = [];
                                        moveSites.forEach(site => {
                                            if (!/^\[/.test(site.url)) {
                                                groupUrlArr.push(site.name);
                                            }
                                        });
                                        cloneSites = [{name: (groupName || "Group") + "-" + event.target.value, url:JSON.stringify(groupUrlArr)}];
                                    } else {
                                        cloneSites = [];
                                        moveSites.forEach(site => {
                                            if (!/^\[/.test(site.url)) {
                                                cloneSites.push({name: site.name + "-" + event.target.value, url: JSON.stringify([site.name])});
                                            }
                                        });
                                    }
                                    window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                                        if (event.target.value === data.type) {
                                            cloneSites.forEach(cloneSite => {
                                                let findIndex = data.sites.findIndex(site => {return site.url === cloneSite.url});
                                                if (findIndex === -1) data.sites.push(cloneSite);
                                            });
                                        }
                                        return data;
                                    });
                                    this.setState(prevState => ({
                                        isOpenSiteEdit: false,
                                        checkeds: Array(prevState.checkeds.length).fill(false)
                                    }));
                                } else {
                                    if (event.target.value === this.state.data.type) return;
                                    if (!window.confirm(window.i18n('moveToConfirm', event.target.value))) return;
                                    let newSites = this.state.data.sites.filter((site, i) => {
                                        return (this.state.checkeds[i] !== true);
                                    })
                                    let newType = {...window.searchData.sitesConfig[this.index], sites: newSites};
                                    window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                                        if (this.state.data.type === data.type) {
                                            return newType;
                                        } else if (event.target.value === data.type) {
                                            data.sites = data.sites.concat(moveSites);
                                        }
                                        return data;
                                    });
                                    this.setState(prevState => ({
                                        data: newType,
                                        isOpenSiteEdit: false,
                                        checkeds: Array(newSites.length).fill(false)
                                    }));
                                }
                                saveConfigToScript();
                            }}
                            autoWidth
                            label={window.i18n('moveTo')}
                        >
                        <MenuItem value={0}>
                            {window.i18n('category')}
                        </MenuItem>
                        {window.searchData.sitesConfig.map((data, index) =>
                            <MenuItem key={data.type} value={data.type}>
                                {data.type}
                            </MenuItem>
                        )}
                        </Select>
                    </FormControl>
                    <FormControlLabel sx={{ ml: 0 }} control={
                        <Checkbox 
                            onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                                this.setState(prevState => ({
                                    cloneSite: event.target.checked
                                }))
                            }}
                            checked={this.state.cloneSite} 
                        />
                    } label={window.i18n('clone')} />
                </Box>

                <ChildSiteIcons 
                    sites={this.state.data.sites} 
                    checkChange={(event: React.ChangeEvent<HTMLInputElement>, i) => {
                        let value = event.target.checked;
                        this.setState(prevState => {
                            let newCheckeds = prevState.checkeds;
                            newCheckeds[i] = value;
                            return {
                                checkeds: newCheckeds
                            }
                        })
                    }}
                    tooLong={this.tooLong}
                    changeSitePos={this.changeSitePos}
                    checkeds={this.state.checkeds}
                    openSiteEdit={this.openSiteEdit}
                />
                <Dialog open={this.state.isOpenSiteEdit} onClose={() => this.closeSiteEdit(false)}>
                    <DialogTitle>{window.i18n(this.state.currentSite.url === '' ? 'addSite' : 'editSite')}</DialogTitle>
                    <DialogContent>
                        <TextField
                            autoFocus
                            margin="dense"
                            id="name"
                            label={window.i18n('siteName')}
                            type="text"
                            fullWidth
                            variant="standard"
                            value={this.state.currentSite.name}
                            onChange={e => {
                                this.setState(prevState => ({
                                    currentSite: {...this.state.currentSite, name: e.target.value}
                                }));
                            }}
                        />
                        <TextField
                            margin="dense"
                            id="url"
                            label={window.i18n('siteUrl')}
                            type="text"
                            fullWidth
                            multiline
                            maxRows={5}
                            variant="standard"
                            value={this.state.currentSite.url}
                            onChange={e => {
                                this.setState(prevState => ({
                                    currentSite: {...this.state.currentSite, url: e.target.value}
                                }));
                            }}
                            placeholder="https://www.google.com/search?q=%s"
                            inputProps={{
                                style: {
                                  resize: 'auto'
                                },
                                spellCheck: 'false'
                            }}
                        />
                        <DialogContentText>
                            {window.i18n('siteUrlTips')}
                        </DialogContentText>
                        <Accordion sx={{margin: '0 -16px!important'}}>
                            <AccordionSummary
                              sx={{background: '#d1d1d120', minHeight: '45px!important', maxHeight: '45px!important'}}
                              expandIcon={<ExpandMoreIcon />}>
                              <Typography align="center" sx={{width: '100%'}}>{window.i18n('moreOptions')}</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                                <TextField
                                    margin="dense"
                                    id="description"
                                    label={window.i18n('description')}
                                    type="text"
                                    fullWidth
                                    multiline
                                    variant="standard"
                                    value={this.state.currentSite.description}
                                    onChange={e => {
                                        this.setState(prevState => ({
                                            currentSite: {...this.state.currentSite, description: e.target.value}
                                        }));
                                    }}
                                    inputProps={{
                                        style: {
                                          resize: 'auto'
                                        }
                                    }}
                                />
                                <TextField
                                    margin="dense"
                                    id="icon"
                                    label={window.i18n('siteIcon')}
                                    type="text"
                                    fullWidth
                                    multiline
                                    maxRows={5}
                                    variant="standard"
                                    value={this.state.currentSite.icon}
                                    onChange={e => {
                                        this.setState(prevState => ({
                                            currentSite: {...this.state.currentSite, icon: e.target.value}
                                        }));
                                    }}
                                    InputProps={{
                                      endAdornment: (
                                          <InputAdornment position="end">
                                            <input
                                                accept="image/*"
                                                style={{ display: "none" }}
                                                id="upload-site-icon"
                                                type="file"
                                                onChange={event => {
                                                    let self = this;
                                                    let file = event.target.files && event.target.files[0];
                                                    if (!file) return;
                                                    if (file.size > 51200 && !window.confirm(window.i18n('imgTooBig'))) {
                                                        event.target.value = "";
                                                        return;
                                                    }
                                                    let reader = new FileReader();
                                                    reader.readAsDataURL(file);
                                                    reader.onload = function() {
                                                        self.setState(prevState => ({
                                                            currentSite: {...self.state.currentSite, icon: reader.result}
                                                        }));
                                                    };
                                                }}
                                            />
                                            <label htmlFor="upload-site-icon">
                                                <IconButton
                                                  edge="end"
                                                  component="span"
                                                >
                                                    <FileUploadIcon/>
                                                </IconButton>
                                            </label>
                                          </InputAdornment>
                                        ),
                                      inputProps: {
                                          style: {
                                              resize: 'auto'
                                          },
                                          spellCheck: 'false'
                                      }
                                    }}
                                />
                                <TextField
                                    margin="dense"
                                    id="keywords"
                                    label={window.i18n('siteKeywords')}
                                    type="text"
                                    fullWidth
                                    multiline
                                    maxRows={5}
                                    variant="standard"
                                    placeholder="wd|q"
                                    value={this.state.currentSite.keywords}
                                    onChange={e => {
                                        this.setState(prevState => ({
                                            currentSite: {...this.state.currentSite, keywords: e.target.value}
                                        }));
                                    }}
                                    inputProps={{ spellCheck: 'false' }}
                                />
                                <DialogContentText>
                                    {window.i18n('keywordRegTips')}
                                </DialogContentText>
                                <TextField
                                    margin="dense"
                                    id="kwFilter"
                                    label={window.i18n('kwFilter')}
                                    type="text"
                                    fullWidth
                                    variant="standard"
                                    placeholder="^\d+$"
                                    value={this.state.currentSite.kwFilter}
                                    onChange={e => {
                                        this.setState(prevState => ({
                                            currentSite: {...this.state.currentSite, kwFilter: e.target.value}
                                        }));
                                    }}
                                    inputProps={{ spellCheck: 'false' }}
                                />
                                <DialogContentText>
                                    {window.i18n('kwFilterTips')}
                                </DialogContentText>
                                <TextField
                                    margin="dense"
                                    id="match"
                                    label={window.i18n('siteMatch')}
                                    type="text"
                                    fullWidth
                                    variant="standard"
                                    placeholder="\.google\.(com|co.jp)"
                                    value={this.state.currentSite.match}
                                    onChange={e => {
                                        this.setState(prevState => ({
                                            currentSite: {...this.state.currentSite, match: e.target.value}
                                        }));
                                    }}
                                    inputProps={{ spellCheck: 'false' }}
                                />
                                <DialogContentText>
                                    {window.i18n('siteMatchTips')}
                                </DialogContentText>
                                <Box sx={{ flexGrow: 1, display: 'flex'}}>
                                    <FormControl sx={{ m: 1, minWidth: 80 }}>
                                        <FormControlLabel
                                            control={
                                                <Switch 
                                                    checked={this.state.currentSite.hideNotMatch} 
                                                    name="hideNotMatch"
                                                    onClick={e => {
                                                        this.setState(prevState => ({
                                                            currentSite: {...prevState.currentSite, hideNotMatch: e.target.checked}
                                                        }));
                                                    }}
                                                />
                                            }
                                            label={window.i18n('hideNotMatch')}
                                        />
                                    </FormControl>
                                    <FormControl sx={{ m: 1, minWidth: 80 }}>
                                        <FormControlLabel
                                            control={
                                                <Switch 
                                                    checked={this.state.currentSite.nobatch} 
                                                    name="nobatch"
                                                    onClick={e => {
                                                        this.setState(prevState => ({
                                                            currentSite: {...prevState.currentSite, nobatch: e.target.checked}
                                                        }));
                                                    }}
                                                />
                                            }
                                            label={window.i18n('nobatch')}
                                        />
                                    </FormControl>
                                </Box>
                                <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'nowrap', mb: 1}}>
                                    <TextField
                                        sx={{ minWidth: 100, maxWidth: 150 }}
                                        margin="dense"
                                        id="match"
                                        label={window.i18n('siteShotcut')}
                                        type="text"
                                        variant="outlined"
                                        value={(this.state.currentSite.shortcut || "").replace(/Key|Digit/, "").replace(/Backquote/, "`").replace(/Minus/, "-").replace(/Equal/, "=").replace(/ArrowUp/, "↑").replace(/ArrowDown/, "↓").replace(/ArrowLeft/, "←").replace(/ArrowRight/, "→")}
                                        inputProps={{ readOnly: 'readonly' }}
                                        onKeyDown={e => {
                                            if (/^(Control|Alt|Meta|Shift)/.test(e.key)) {
                                                return;
                                            }
                                            this.setState(prevState => ({
                                                currentSite: {
                                                    ...this.state.currentSite,
                                                    ctrl: e.ctrlKey,
                                                    alt: e.altKey,
                                                    shift: e.shiftKey,
                                                    meta: e.metaKey,
                                                    shortcut: (e.key === 'Escape' || e.key === 'Backspace') ? '' : (e.code || e.key)
                                                }
                                            }));
                                        }}
                                    />
                                    <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'wrap'}}>
                                        <FormControl sx={{ minWidth: 60 }}>
                                            <FormControlLabel className="keyboardBtn"
                                                control={
                                                    <Switch 
                                                        checked={this.state.currentSite.ctrl} 
                                                        name="ctrl"
                                                        onClick={e => {
                                                            this.setState(prevState => ({
                                                                currentSite: {...prevState.currentSite, ctrl: e.target.checked}
                                                            }));
                                                        }}
                                                    />
                                                }
                                                label='Ctrl'
                                                labelPlacement="bottom"
                                            />
                                        </FormControl>
                                        <FormControl sx={{ minWidth: 60 }}>
                                            <FormControlLabel className="keyboardBtn"
                                                control={
                                                    <Switch 
                                                        checked={this.state.currentSite.alt} 
                                                        name="alt"
                                                        onClick={e => {
                                                            this.setState(prevState => ({
                                                                currentSite: {...prevState.currentSite, alt: e.target.checked}
                                                            }));
                                                        }}
                                                    />
                                                }
                                                label='Alt'
                                                labelPlacement="bottom"
                                            />
                                        </FormControl>
                                        <FormControl sx={{ minWidth: 60 }}>
                                            <FormControlLabel className="keyboardBtn"
                                                control={
                                                    <Switch 
                                                        checked={this.state.currentSite.shift} 
                                                        name="shift"
                                                        onClick={e => {
                                                            this.setState(prevState => ({
                                                                currentSite: {...prevState.currentSite, shift: e.target.checked}
                                                            }));
                                                        }}
                                                    />
                                                }
                                                label='Shift'
                                                labelPlacement="bottom"
                                            />
                                        </FormControl>
                                        <FormControl sx={{ minWidth: 60 }}>
                                            <FormControlLabel className="keyboardBtn"
                                                control={
                                                    <Switch 
                                                        checked={this.state.currentSite.meta} 
                                                        name="meta"
                                                        onClick={e => {
                                                            this.setState(prevState => ({
                                                                currentSite: {...prevState.currentSite, meta: e.target.checked}
                                                            }));
                                                        }}
                                                    />
                                                }
                                                label='Meta'
                                                labelPlacement="bottom"
                                            />
                                        </FormControl>
                                    </Box>
                                </Box>
                                <Box sx={{flexGrow: 1, display: 'flex', flexWrap: 'nowrap'}}>
                                    <FormControl sx={{ minWidth: '30%' }}>
                                        <InputLabel>{window.i18n('openSelect')}</InputLabel>
                                        <Select
                                            value={this.state.currentSite.openInNewTab}
                                            name="openInNewTab"
                                            onChange={(e: SelectChangeEvent) => {
                                                this.setState(prevState => ({
                                                    currentSite: {...this.state.currentSite, openInNewTab: e.target.value}
                                                }));
                                            }}
                                            autoWidth
                                            label={window.i18n('openSelect')}
                                        >
                                            <MenuItem value={-1}>{window.i18n("openInDefaultOption")}</MenuItem>
                                            <MenuItem value={1}>{window.i18n("openInNewTabOption")}</MenuItem>
                                            <MenuItem value={0}>{window.i18n("openInCurrentOption")}</MenuItem>
                                            <MenuItem value={4}>{window.i18n("openInBackOption")}</MenuItem>
                                            <MenuItem value={2}>{window.i18n("openInIncognitoOption")}</MenuItem>
                                            <MenuItem value={3}>{window.i18n("openInMinWindowOption")}</MenuItem>
                                        </Select>
                                    </FormControl>
                                    <Autocomplete
                                        disablePortal
                                        margin="dense"
                                        sx={{ ml: 1 }}
                                        id="charset"
                                        fullWidth
                                        variant="standard"
                                        options={allCharset}
                                        value={this.state.currentSite.charset}
                                        onChange={e => {
                                            this.setState(prevState => ({
                                                currentSite: {...this.state.currentSite, charset: e.target.textContent}
                                            }));
                                        }}
                                        renderInput={(params) => <TextField 
                                            {...params}
                                            label={window.i18n('siteCharset')} 
                                        />}
                                    />
                                </Box>
                            </AccordionDetails>
                        </Accordion>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => this.setState(prevState => ({ isOpenLocalApp: true }))}>{window.i18n('localAppAddBtn')}</Button>
                        <Button variant="outlined" style={{display:this.editSite?'':'none'}} color="error" startIcon={<DeleteIcon />} onClick={this.handleDeleteSite}>{window.i18n('delete')}</Button>
                        <Button onClick={() => this.closeSiteEdit(false)}>{window.i18n('cancel')}</Button>
                        <Button onClick={() => this.closeSiteEdit(true)}>{window.i18n(this.state.currentSite.url === '' ? 'add' : 'save')}</Button>
                    </DialogActions>
                </Dialog>
                <Dialog open={this.state.isOpenSiteEdit&&this.state.isOpenLocalApp} onClose={() => this.setState(prevState => ({ isOpenLocalApp: false }))}>
                    <DialogTitle>{window.i18n('localApp')}</DialogTitle>
                    <DialogContent>
                        <TextField
                            autoFocus
                            margin="dense"
                            id="localAppCall"
                            label={window.i18n('localAppCall')}
                            type="text"
                            fullWidth
                            variant="standard"
                            placeholder={'"C:\\Program Files\\MPV\\mpv.exe" --stream %u'}
                            inputProps={{ spellCheck: 'false' }}
                        />
                        <TextField
                            margin="dense"
                            id="localAppName"
                            label={window.i18n('localAppName')}
                            type="text"
                            fullWidth
                            inputProps={{ maxLength: 5, spellCheck: 'false' }} 
                            variant="standard"
                        />
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => this.setState(prevState => ({ isOpenLocalApp: false }))}>{window.i18n('cancel')}</Button>
                        <Button onClick={() => {
                            let localAppCall = document.querySelector("#localAppCall");
                            let localAppName = document.querySelector("#localAppName");
                            let n = localAppName.value;
                            let c = localAppCall.value.replace(/([^\\])[\\]([^\\])/g, "$1\\\\$2");
                            let match = c.match(/^"((([a-z]:).+?)([^/\\]+))" (.*(%.+?)\b.*)/i);
                            if (!match) {
                                match = c.match(/^((([a-z]:)\S+?)([^/\\ ]+)) (.*(%.+?)\b.*)/i);
                            }
                            if (!match) {
                                return this.handleAlertOpen(window.i18n('localAppUnknowCall'));
                            }
                            if(!n) {
                                n = match[4].replace(/([a-z]+).*/i, "$1")
                            }
                            if(!/^\w+$/.test(n)){
                                return this.handleAlertOpen(window.i18n('localAppWrongName'));
                            }
                            n = "SearchJumper-" + n;
                            this.setState(prevState => ({
                                currentSite: {...this.state.currentSite, url: n + "://" + match[6]}
                            }));
                            let blobStr = [`
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\\${n}]
@="URL:${n} Protocol"
"URL Protocol"=""

[HKEY_CLASSES_ROOT\\${n}\\DefaultIcon]
@="\\"${match[1]}\\",1"

[HKEY_CLASSES_ROOT\\${n}\\shell]

[HKEY_CLASSES_ROOT\\${n}\\shell\\open]

[HKEY_CLASSES_ROOT\\${n}\\shell\\open\\command]
@="cmd /c set m=%1 & call set m=%%m:${n}://=%% & ${match[3]} & cd \\"${match[2]}\\" & call ${match[4]} ${match[5].replace(match[6], '%%m%%')}"
`.trim()];
                            let myBlob = new Blob(blobStr, { type: "application/text" });
                            this.downloadEle.download = `${n}.reg`;
                            this.downloadEle.href = window.URL.createObjectURL(myBlob);
                            this.downloadEle.click();
                            this.setState(prevState => ({ 
                                isOpenLocalApp: false 
                            }));
                        }}>{window.i18n('geneRegFile')}</Button>
                    </DialogActions>
                </Dialog>
            </Box>
        );
    }
}

const allCharset = [
  "","gbk","gb18030","big5","big5-hkscs","utf-8","utf-16le","shift-jis","euc-jp","iso-2022-jp","euc-kr","iso-2022-kr","macintosh","koi8-r","koi8-u",
    "windows-1250","windows-1251","windows-1252","windows-1253","windows-1254","windows-1255","windows-1256","windows-1257","windows-1258",
    "iso-8859-1","iso-8859-2","iso-8859-3","iso-8859-4","iso-8859-5","iso-8859-6","iso-8859-7","iso-8859-8","iso-8859-8-i","iso-8859-9","iso-8859-10","iso-8859-11","iso-8859-13","iso-8859-14","iso-8859-15","iso-8859-16"
];

function a11yProps(index: number) {
    return {
        id: `vertical-tab-${index}`,
        'aria-controls': `vertical-tabpanel-${index}`,
    };
}

function typeObject(obj) {
    obj = obj || {};
    let openInNewTab = "";
    if (typeof obj.openInNewTab === 'undefined') {
        openInNewTab = -1;
    } else {
        openInNewTab = obj.openInNewTab;
        if (openInNewTab === true) {
            openInNewTab = 1;
        } else if (openInNewTab === false) {
            openInNewTab = 0;
        }
    }
    return {
        type: obj.type || '',
        icon: obj.icon || '',
        match: obj.match || '',
        description: obj.description || '',
        selectTxt: obj.selectTxt || false,
        selectImg: obj.selectImg || false,
        selectAudio: obj.selectAudio || false,
        selectVideo: obj.selectVideo || false,
        selectLink: obj.selectLink || false,
        selectPage: obj.selectPage || false,
        openInNewTab: openInNewTab,
        shortcut: obj.shortcut || '',
        ctrl: obj.ctrl || false,
        alt: obj.alt || false,
        shift: obj.shift || false,
        meta: obj.meta || false,
    };
}

function siteObject(obj) {
    obj = obj || {};
    let openInNewTab = "";
    if (typeof obj.openInNewTab === 'undefined') {
        openInNewTab = -1;
    } else {
        openInNewTab = obj.openInNewTab;
        if (openInNewTab === true) {
            openInNewTab = 1;
        } else if (openInNewTab === false) {
            openInNewTab = 0;
        }
    }
    return {
        name: obj.name || '',
        url: obj.url || '',
        icon: obj.icon || '',
        keywords: obj.keywords || '',
        kwFilter: obj.kwFilter || '',
        description: obj.description || '',
        match: obj.match || '',
        charset: obj.charset || '',
        shortcut: obj.shortcut || '',
        ctrl: obj.ctrl || false,
        alt: obj.alt || false,
        shift: obj.shift || false,
        meta: obj.meta || false,
        nobatch: obj.nobatch || false,
        openInNewTab: openInNewTab,
        hideNotMatch: obj.hideNotMatch || false
    };
}

function getOffsetLeft(ele) {
    var actualLeft = ele.offsetLeft;
    var current = ele.offsetParent;
    while (current) {
        actualLeft += current.offsetLeft;
        current = current.offsetParent;
    }
    return actualLeft;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`vertical-tabpanel-${index}`}
            aria-labelledby={`vertical-tab-${index}`}
            style={{width: '100%'}}
            {...other}
        >
            {value === index && (
                <Box>
                    {children}
                </Box>
            )}
        </div>
    );
}

function createData(
  param: string,
  info: string
) {
  return { param, info };
}

var verifyArray, filterDataListTimer, filterHighlightTimer, inited = false;

function sendVerifyRequest() {
    if (!verifyArray || !verifyArray.length) {
        let verifyResultList = document.getElementById('verifyResultList');
        if (verifyResultList) verifyResultList.classList.add('finished');
        let verifyDelBtn = document.getElementById('verifyDelBtn');
        if (verifyDelBtn) verifyDelBtn.classList.add('finished');
        let verifyStatus = document.getElementById('verifyStatus');
        if (verifyStatus) verifyStatus.innerText = window.i18n('verifyFinish');
        return;
    }
    let item = verifyArray.shift();
    if (item) {
        var saveMessage = new CustomEvent('verifyUrl', {
            detail: {
                url: item.url.replace(/#.*/, '').replace(/%[pn].*/, '').replace(/%s[lure]?\b(\.replace\(.*?\))*/g, 'searchJumper').replace(/%[ut]\b/i, 'https://google.com'),
                name: item.name
            }
        });
        document.dispatchEvent(saveMessage);
    } else {
        return;
    }
}

function forwordToSite(inputWord) {
    let filterEngine = document.querySelector('.site-icon.filter');
    if (filterEngine) {
        filterEngine.classList.remove('filter');
    }
    clearTimeout(filterDataListTimer);
    clearTimeout(filterHighlightTimer);
    if (!inputWord) return true;
    let filterEngineName = "";
    let filterGroup = false;
    if (inputWord.indexOf(`【${window.i18n('category')}】`) === 0) {
        filterGroup = true;
        inputWord = inputWord.replace(`【${window.i18n('category')}】`, '');
    }
    let typeIndex = window.searchData.sitesConfig.findIndex((data, index) => {
        if (filterGroup) return inputWord === data.type;
        return data.sites.findIndex((site, i) => {
            if (site.name === inputWord || site.url.replace(/\n/g, "") === inputWord) {
                filterEngineName = site.name;
                return true;
            }
            return false;
        }) > -1;
    });
    if (typeIndex > -1) {
        if (filterGroup) return typeIndex;
        filterHighlightTimer = setInterval(() => {
            [].every.call(document.querySelectorAll(".site-icon"), icon => {
                if (icon.childNodes[1].title === filterEngineName) {
                    icon.classList.add("filter");
                    icon.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
                    clearTimeout(filterHighlightTimer);
                    return false;
                }
                return true;
            });
        }, 500);
        return typeIndex;
    }
    return false;
}

export default function Engines() {
    const rows = [
      createData('%s', window.i18n('param_s')),
      createData('%sl', window.i18n('param_sl')),
      createData('%su', window.i18n('param_su')),
      createData('%sr', window.i18n('param_sr')),
      createData('%se', window.i18n('param_se')),
      createData('%S', window.i18n('param_S')),
      createData('%u', window.i18n('param_u')),
      createData('%h', window.i18n('param_h')),
      createData('%t', window.i18n('param_t')),
      createData('%b', window.i18n('param_b')),
      createData('%n', window.i18n('param_n')),
      createData('%i', window.i18n('param_i')),
      createData('%e', window.i18n('param_e')),
      createData('%c', window.i18n('param_c')),
      createData('%s.replace', window.i18n('param_sre')),
      createData('%s[]', window.i18n('param_ssplit')),
      createData('%p{params}', window.i18n('param_p1')),
      createData('%P{params}', window.i18n('param_p2')),
      createData('#p{params}', window.i18n('param_p3')),
      createData('["siteName1","siteName2"]', window.i18n('param_group')),
      createData('%input{tips}', window.i18n('param_input')),
      createData('%element{}', window.i18n('param_ele')),
      createData('%date', window.i18n('param_date1')),
      createData('%date{}', window.i18n('param_date2')),
      createData('%element{}.prop()', window.i18n('param_elep')),
      createData('%element{}.replace()', window.i18n('param_elere')),
      createData('copy', window.i18n('param_cp')),
      createData('paste', window.i18n('param_paste')),
      createData('showTips', window.i18n('param_showTips')),
      createData('find', window.i18n('param_find')),
      createData('find.addto()', window.i18n('param_findadd'))      
    ];
    if (/^(http|ftp)/i.test(window.location.protocol)) {
        if (window.lang.indexOf("zh") === 0) {
            rows.splice(4, 0, createData('%ss', window.i18n('param_ss')), createData('%st', window.i18n('param_st')));
        }
        rows.push(createData('javascript', window.i18n('javascript')));
    }
    let selectTxt = -1, selectImg = -1, selectLink = -1, selectPage = -1, selectAll = -1;
    for (let i = 0; i < window.searchData.sitesConfig.length; i++) {
        let site = window.searchData.sitesConfig[i];
        if (site.match) continue;
        if (selectAll === -1 && site.selectTxt && site.selectImg && site.selectAudio && site.selectVideo && site.selectLink && site.selectPage) {
            selectAll = i;
            continue;
        }

        if (selectTxt === -1 && site.selectTxt) {
            selectTxt = i;
        }
        if (selectImg === -1 && site.selectImg) {
            selectImg = i;
        }
        if (selectLink === -1 && site.selectLink) {
            selectLink = i;
        }
        if (selectPage === -1 && site.selectPage) {
            selectPage = i;
        }
        if (selectTxt !== -1 && selectImg !== -1 && selectLink !== -1 && selectPage !== -1) break;
    }
    if (selectAll !== -1) {
        if (selectTxt === -1) selectTxt = selectAll;
        if (selectImg === -1) selectImg = selectAll;
        if (selectLink === -1) selectLink = selectAll;
        if (selectPage === -1) selectPage = selectAll;
    }
    const [value, setValue] = React.useState(0);

    const [editTypeOpen, setTypeOpen] = React.useState(false);
    const [editTypeData, setTypeData] = React.useState(typeObject(false));

    const [alertBody, setAlert] = React.useState({openAlert: false, alertContent: '', alertType: 'error'});

    const [refresh, setRefresh] = React.useState(false);
     
    React.useEffect(() => {
        if (inited) return;
        inited = true;
        window.addEventListener('message',function(e){
          if (e.data.command === 'refresh') {
            setRefresh(true);
          } else if (e.data.command === 'verifyResult') {
            let verifyResultList = document.body.querySelector('#verifyResultList');
            if (!verifyResultList) return;
            let verifyItem = document.createElement('tr');
            let statusStr = '';
            switch(e.data.status) {
                case 200:
                    statusStr = 'OK';
                    break;
                case 300:
                    statusStr = 'Multiple Choices';
                    break;
                case 301:
                    statusStr = 'Moved Permanently';
                    break;
                case 304:
                    statusStr = 'Not Modified';
                    break;
                case 400:
                    statusStr = 'Bad Request';
                    break;
                case 401:
                    statusStr = 'Unauthorized';
                    break;
                case 403:
                    statusStr = 'Forbidden';
                    break;
                case 404:
                    statusStr = 'Not Found';
                    break;
                case 500:
                    statusStr = 'Internal Server Error';
                    break;
                case 502:
                    statusStr = 'Bad Gateway';
                    break;
                case 503:
                    statusStr = 'Service Unavailable';
                    break;
                case 504:
                    statusStr = 'Gateway Timeout';
                    break;
                default:
                    break;
            }
            verifyItem.innerHTML = `<td>🖱${e.data.name}</td><td><a target="_blank" href='${e.data.url}'>${e.data.url}</a></td><td><span ${e.data.status < 300 ? '' : 'style = "color: red"'} title='${statusStr}'>${e.data.status}<input type="checkbox" data-name="${e.data.name}" checked/></span></td>`;
            if (e.data.status < 300) verifyItem.classList.add('okItem');
            verifyItem.onclick = ev => {
                if (ev.target.tagName !== 'A') {
                    let typeIndex = forwordToSite(e.data.name);
                    if (typeIndex !== false) {
                        setValue(typeIndex);
                    }
                }
            };
            verifyResultList.appendChild(verifyItem);
            sendVerifyRequest();
          }
        });
    }, []);
    React.useEffect(() => {
        refresh && setTimeout(() => setRefresh(false), 0)
    }, [refresh]);

    const handleChange = (event: React.SyntheticEvent, newValue: number) => {
        setValue(newValue);
    };

    const openTypeEdit = editType => {
        if (editType !== false) {
            editType = window.searchData.sitesConfig[editType];
        }
        setTypeData(typeObject(editType));
        setTypeOpen(true);
    };

    const getMinTypeData = (typeData) => {
        let minType = {};
        if (typeData.type) {
            minType.type = typeData.type;
        }
        minType.icon = typeData.icon || '';
        if (typeData.match) {
            minType.match = typeData.match;
        }
        if (typeData.description) {
            minType.description = typeData.description;
        }
        if (typeData.selectTxt) {
            minType.selectTxt = typeData.selectTxt;
        }
        if (typeData.selectImg) {
            minType.selectImg = typeData.selectImg;
        }
        if (typeData.selectAudio) {
            minType.selectAudio = typeData.selectAudio;
        }
        if (typeData.selectVideo) {
            minType.selectVideo = typeData.selectVideo;
        }
        if (typeData.selectLink) {
            minType.selectLink = typeData.selectLink;
        }
        if (typeData.selectPage) {
            minType.selectPage = typeData.selectPage;
        }
        if (typeData.openInNewTab !== -1) {
            minType.openInNewTab = typeData.openInNewTab;
        }
        if (typeData.shortcut) {
            minType.shortcut = typeData.shortcut;
        }
        if (typeData.ctrl) {
            minType.ctrl = typeData.ctrl;
        }
        if (typeData.alt) {
            minType.alt = typeData.alt;
        }
        if (typeData.shift) {
            minType.shift = typeData.shift;
        }
        if (typeData.meta) {
            minType.meta = typeData.meta;
        }
        minType.sites = typeData.sites;
        return minType;
    };

    const changeType = (newType) => {
        let newData;
        if (editTypeData.type === '') {
            if (!newType || newType.type === '') return;
            newData = window.searchData.sitesConfig.concat([{...getMinTypeData(newType), sites: []}])
        } else {
            if (newType === false) {
                newData = window.searchData.sitesConfig.filter(data =>{
                    return (editTypeData.type !== data.type)
                });
                let newValue = value - 1;
                if (newValue < 0) newValue = 0;
                setValue(newValue);
            } else {
                newData = window.searchData.sitesConfig.map(data =>{
                    if (editTypeData.type === data.type) {
                        newType.sites=data.sites;
                        return getMinTypeData(newType);
                    }
                    return data;
                });
                setTypeData(newType);
            }
        }
        window.searchData.sitesConfig=newData;
        saveConfigToScript();
    };

    const handleDeleteType = () => {
        changeType(false);
        setTypeOpen(false);
    };

    const handleAlertOpen = (content, type) => {
        switch (type) {
            case 0:
                type = "error";
            break;
            case 1:
                type = "warning";
            break;
            case 2:
                type = "info";
            break;
            case 3:
                type = "success";
            break;
            default:
                type = "error";
            break;
        }
        setAlert({
            openAlert: true,
            alertContent: content,
            alertType: type
        });
    };

    const handleAlertClose = () => {
        setAlert({
            openAlert: false,
            alertContent: '',
            alertType: alertBody.alertType
        });
    };

    const changeTypePos = (targetType, event) => {
        hideDragLine();
        let dragType;
        try {
            dragType = JSON.parse(event.dataTransfer.getData("data"));
        } catch (e) {
            console.log(e);
            return;
        }
        let target = event.currentTarget;
        let isRight = event.clientX > getOffsetLeft(target) + target.offsetWidth / 2;
        if (!dragType.type) {
            if (!dragType.url) return;
            for (let i = 0; i < targetType.sites.length; i++) {
                if (targetType.sites[i].url === dragType.url) return;
            }
            window.searchData.sitesConfig = window.searchData.sitesConfig.map((data, i) =>{
                if (targetType.type === data.type) {
                    data.sites = data.sites.concat([dragType]);
                } else if (value === i) {
                    data.sites = data.sites.filter(site => {
                        return (site.url !== dragType.url);
                    })
                }
                return data;
            });
            saveConfigToScript();
            setRefresh(true);
            return;
        }
        if (targetType.type === dragType.type) return;
        let newTypes = window.searchData.sitesConfig.filter(typeData => {
            return (typeData.type !== dragType.type);
        })
        for (let i = 0; i < newTypes.length; i++) {
            if (newTypes[i].type === targetType.type) {
                newTypes.splice((isRight ? i + 1 : i), 0, dragType);
                break;
            }
        }
        window.searchData.sitesConfig = newTypes;
        saveConfigToScript();
        setRefresh(true);
    };


    const dragOver = e => {
        e.preventDefault();
        if (!dragTargetLine) dragTargetLine = document.querySelector(`#dragTargetLine`);
        if (dragTargetLine) {
            dragTargetLine.style.display = "block";
            let target = e.currentTarget;
            target.parentNode.parentNode.appendChild(dragTargetLine);
            let isRight = e.clientX > getOffsetLeft(target) + target.offsetWidth / 2;
            dragTargetLine.style.top = target.offsetTop + "px";
            dragTargetLine.style.left = (isRight ? target.offsetLeft + target.offsetWidth : target.offsetLeft) + "px";
        }
    };

    return (
        <Box sx={{pb: 3}}>
            <Paper elevation={5} sx={{textAlign:'center', borderRadius:'10px'}}>
                <h2 style={{padding:'5px'}}>{window.i18n('searchEngines')}</h2>
            </Paper>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', flexGrow: 1, display: 'flex' }}>
                <Tabs value={value} onChange={handleChange} aria-label="types" variant="scrollable" scrollButtons allowScrollButtonsMobile>
                    {
                        window.searchData.sitesConfig.map((data, index) =>
                            <Tab  
                                sx={{width: 65}}
                                className={(selectTxt === index ? 'selectTxt ' : '') + 
                                            (selectImg === index ? 'selectImg ' : '') +
                                            (selectLink === index ? 'selectLink ' : '') +
                                            (selectPage === index ? 'selectPage ' : '') +
                                            (data.match === '0' ? 'hideIcon' : '')}
                                draggable='true'
                                onDrop={e => {changeTypePos(data, e)}} 
                                onDragStart={e => {e.dataTransfer.setData("data", JSON.stringify(data))}} 
                                onDragOver={e => {dragOver(e)}} 
                                onDragLeave={e => {hideDragLine()}}
                                icon={
                                    /^(http|data:)/.test(data.icon)?(
                                        <img alt={data.type} src={(/^http/.test(data.icon) && window.cacheIcon[data.icon]) || data.icon} style={{m:1, background: 'darkgray', borderRadius: '35px', width: '65px', height: '65px', padding: '15px', boxSizing: 'border-box', overflow: 'hidden'}} />
                                    ):(
                                        <i style={{background: 'darkgray', lineHeight: '65px', width: '65px', height: '65px', fontSize: '30px', color: 'white', borderRadius: '35px', overflow: 'hidden'}} className={`${/^fa/.test(data.icon) ? data.icon : "fa fa-" + data.icon}`}>{data.icon ? '' : data.type}</i>
                                    )} 
                                label={data.type.slice(0, 10)} 
                                title={data.description || data.type}
                                key={index} 
                                {...a11yProps(index)} 
                            />
                        )
                    }
                </Tabs>
                <Box sx={{ flexDirection: 'column', display: 'flex' }}>
                    <IconButton color="primary" sx={{mt: '8px'}} onClick={() => {openTypeEdit(false)}}>
                        <AddCircleOutlineIcon sx={{fontSize: '30px'}}/>
                    </IconButton>
                    <IconButton color="primary" onClick={(e) => {
                        let scrollCon = e.currentTarget.parentNode.parentNode;
                        scrollCon.classList.toggle('unfold');
                        scrollCon.querySelector(".MuiTabs-scroller").scrollLeft += 1;
                    }}>
                        <FullscreenIcon sx={{fontSize: '30px'}}/>
                    </IconButton>
                </Box>
            </Box>
            {window.searchData.sitesConfig.map((data, index) =>
                <TabPanel
                    value={value}
                    index={index}
                    key={data.type}
                    className={(selectTxt === index ? 'selectTxt ' : '') + 
                                (selectImg === index ? 'selectImg ' : '') +
                                (selectLink === index ? 'selectLink ' : '') +
                                (selectPage === index ? 'selectPage ' : '')}
                >
                    <SitesList data={data} openTypeEdit={openTypeEdit} index={index} handleAlertOpen={handleAlertOpen}/>
                </TabPanel>
            )}
            <TypeEdit 
                typeOpen={editTypeOpen}
                data={editTypeData} 
                handleDeleteType={handleDeleteType} 
                handleAlertOpen={handleAlertOpen}
                changeType={changeType}
                closeHandler={() => {setTypeOpen(false)}}
            />
            <Snackbar open={alertBody.openAlert} autoHideDuration={2000} anchorOrigin={{vertical: 'top', horizontal: 'center'}} onClose={handleAlertClose}>
                <MuiAlert elevation={6} variant="filled" onClose={handleAlertClose} severity={alertBody.alertType} sx={{ width: '100%' }} >
                  {alertBody.alertContent}
                </MuiAlert>
            </Snackbar>
            <Box sx={{ mt:1, textAlign:'center', whiteSpace:'nowrap'}}>
                <input placeholder={window.i18n('filterEngine')} className={'filterEngine'} list="filterlist"
                    onChange={e => {
                        clearTimeout(filterDataListTimer);
                        clearTimeout(filterHighlightTimer);
                        filterDataListTimer = setTimeout(() => {
                            let list = e.target.list;
                            let inputWord = e.target.value, inputWordLc;
                            [].find.call(list.children, option => {
                                if(option.value === inputWord) {
                                    let typeIndex = forwordToSite(inputWord);
                                    if (typeIndex !== false) {
                                        setValue(typeIndex);
                                    }
                                    return true;
                                }
                                return false;
                            });
                            list.innerHTML = "";
                            if (inputWord) inputWordLc = inputWord.toLowerCase();
                            else return;
                            window.searchData.sitesConfig.every((data, index) => {
                                if (data.type.toLowerCase().indexOf(inputWordLc) !== -1) {
                                    if (`【${window.i18n('category')}】` + data.type !== inputWord) {
                                        let option = document.createElement('option');
                                        option.value = `【${window.i18n('category')}】` + data.type;
                                        list.appendChild(option);
                                    }
                                }
                                return data.sites.every((site, i) => {
                                    if (site.name.toLowerCase().indexOf(inputWordLc) !== -1) {
                                        if (site.name !== inputWord) {
                                            let option = document.createElement('option');
                                            option.value = site.name;
                                            list.appendChild(option);
                                        }
                                    } else if (site.url.length < 1000 && site.url.indexOf(inputWordLc) !== -1) {
                                        if (site.url !== inputWord) {
                                            let option = document.createElement('option');
                                            option.value = site.url;
                                            list.appendChild(option);
                                        }
                                    }
                                    return list.children.length < 10;
                                });
                            });
                        }, 500);
                    }}
                    onKeyDown={e => {
                        if (e.key === 'Enter' && e.target.value) {
                            let inputWord = e.target.value;
                            let typeIndex = forwordToSite(inputWord);
                            if (typeIndex !== false) {
                                setValue(typeIndex);
                                let list = e.target.list;
                                if (list) list.innerHTML = "";
                            }
                        }
                    }}
                />
                <SearchIcon/>
                <Button variant="contained" id="verifyBtn" title={window.i18n('verifyBtn')} endIcon={<DomainVerificationIcon sx={{mr: '-4px', mt: '-2px'}}/>}
                    onClick={() => {
                        verifyArray = [];
                        let verifyResultList = document.getElementById('verifyResultList');
                        let verifyPanel = document.getElementById('verifyPanel');
                        let verifyStatus = document.getElementById('verifyStatus');
                        if (!verifyResultList || !verifyPanel || !verifyStatus || !window.searchData || !window.searchData.sitesConfig) return;
                        verifyPanel.style.display = 'block';
                        verifyResultList.classList.remove('finished');
                        let verifyDelBtn = document.getElementById('verifyDelBtn');
                        if (verifyDelBtn) verifyDelBtn.classList.remove('finished');
                        verifyResultList.innerHTML = '';
                        verifyStatus.innerText = window.i18n('verifying');
                        window.searchData.sitesConfig.forEach(data => {
                            data.sites.forEach(site => {
                                if (/^https?:\/\//.test(site.url)) {
                                    verifyArray.push(site);
                                }
                            });
                        });
                        for (var i = 0; i < 5; i++) {
                            sendVerifyRequest();
                        }
                    }}
                ></Button>
                <datalist id="filterlist"></datalist>
            </Box>
            <Paper sx={{ pt: 1, pb: 2, boxShadow: 'unset', textAlign:'center', borderRadius:'3px', overflow: 'auto', whiteSpace: 'nowrap' }}>
                <span
                    className={'selectTxt'}
                    onClick={() => {
                        let selectTxt = document.body.querySelector('.selectTxt');
                        if (selectTxt) {
                            selectTxt.scrollIntoView(false);
                            selectTxt.click();
                        }
                    }}
                >
                    {window.i18n('typeEnableSelTxt')}
                </span>
                <span
                    className={'selectImg'}
                    onClick={() => {
                        let selectImg = document.body.querySelector('.selectImg');
                        if (selectImg) {
                            selectImg.scrollIntoView(false);
                            selectImg.click();
                        }
                    }}
                >
                    {window.i18n('typeEnableSelImg')}
                </span>
                <span
                    className={'selectLink'}
                    onClick={() => {
                        let selectLink = document.body.querySelector('.selectLink');
                        if (selectLink) {
                            selectLink.scrollIntoView(false);
                            selectLink.click();
                        }
                    }}
                >
                    {window.i18n('typeEnableSelLink')}
                </span>
                <span
                    className={'selectPage'}
                    onClick={() => {
                        let selectPage = document.body.querySelector('.selectPage');
                        if (selectPage) {
                            selectPage.scrollIntoView(false);
                            selectPage.click();
                        }
                    }}
                >
                    {window.i18n('typeEnableSelPage')}
                </span>
            </Paper>
            <Accordion id='verifyPanel' defaultExpanded={true} sx={{ boxShadow: 5, maxHeight: '60vh', overflow: 'auto' }}>
                <AccordionSummary
                  sx={{background: '#f9f9f9', position: 'sticky', top: 0, minHeight: '45px!important', maxHeight: '45px!important'}}
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                >
                    <Typography sx={{display: 'block', width: '100%', textAlign: 'center', fontSize: '1.3em', fontWeight: 'bold'}}>
                        {window.i18n("verifyResult")}
                        <span style={{marginLeft: '10px'}} id='verifyStatus'></span>
                        <Button sx={{position: "absolute", right: "50px", marginTop: "-2px"}} variant="contained" color="error" id="verifyDelBtn"
                            onClick={e => {
                                e.preventDefault();
                                e.stopPropagation();
                                if (!window.confirm(window.i18n('deleteConfirm'))) return;
                                let waitForDel = document.body.querySelectorAll("#verifyResultList>tr:not(.okItem) input:checked");
                                if (waitForDel.length) {
                                    [].forEach.call(waitForDel, input => {
                                        let siteName = input.dataset.name;
                                        if (!siteName) return;
                                        let typeIndex = window.searchData.sitesConfig.findIndex((data, index) => {
                                            return data.sites.findIndex((site, i) => {
                                                if (site.name === siteName) {
                                                    return true;
                                                }
                                                return false;
                                            }) > -1;
                                        });
                                        if (typeIndex < 0) return;
                                        let currentType = window.searchData.sitesConfig[typeIndex];
                                        let newSites = currentType.sites.filter(site => {
                                            return (site.name !== siteName);
                                        });
                                        let newType = {...currentType, sites: newSites};
                                        window.searchData.sitesConfig = window.searchData.sitesConfig.map(data =>{
                                            let returnData = data;
                                            if (currentType.type === data.type) {
                                                returnData = newType;
                                            }
                                            returnData.sites = returnData.sites.map(site => {
                                                if (/^\[/.test(site.url)) {
                                                    try {
                                                        let namesArr = JSON.parse(site.url);
                                                        namesArr = namesArr.filter(n => {
                                                            return (n !== siteName);
                                                        });
                                                        site.url = namesArr.length === 0 ? '' : JSON.stringify(namesArr);
                                                    } catch (e) {
                                                        console.log(e);
                                                    }
                                                }
                                                return site;
                                            });
                                            returnData.sites = returnData.sites.filter(site => {
                                                return site.url !== '';
                                            });
                                            return returnData;
                                        });
                                        input.parentNode.parentNode.parentNode.parentNode.removeChild(input.parentNode.parentNode.parentNode);
                                    });
                                    saveConfigToScript();
                                    setRefresh(true);
                                    handleAlertOpen(window.i18n('deleteOk'));
                                }
                            }}
                        >{window.i18n("delete")}</Button>
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <table><tbody id='verifyResultList'></tbody></table>
                </AccordionDetails>
            </Accordion>
            <Accordion defaultExpanded={true} sx={{ boxShadow: 3, maxHeight: '60vh', overflow: 'auto' }}>
                <AccordionSummary
                  sx={{background: '#f9f9f9', position: 'sticky', top: 0, minHeight: '45px!important', maxHeight: '45px!important'}}
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                >
                  <Typography sx={{display: 'block', width: '100%', textAlign: 'center', fontSize: '1.3em', fontWeight: 'bold'}}>{window.i18n("paramTitle")}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Paper
                      component="form"
                      target="_blank"
                      action="https://fontawesome.com/v6/search?m=free"
                      method="get"
                      sx={{ mt: '10px', p: '2px 4px', display: 'flex', alignItems: 'center', border: '1px solid rgba(0, 0, 0, 0.25)', boxShadow: 'unset' }}
                    >
                        <InputBase
                            name="q"
                            sx={{ ml: 1, flex: 1 }}
                            placeholder={window.i18n("searchFontawesome")}
                            inputProps={{ 'aria-label': 'search fontawesome' }}
                        />
                        <IconButton type="submit" sx={{ p: '10px' }} aria-label="search">
                        <SearchIcon />
                        </IconButton>
                        <InputBase
                            name="m"
                            value="free"
                            type="hidden"
                        />
                    </Paper>
                    <Paper
                      component="form"
                      target="_blank"
                      action="https://mycroftproject.com/search-engines.html"
                      method="get"
                      sx={{ mt: '10px', p: '2px 4px', display: 'flex', alignItems: 'center', border: '1px solid rgba(0, 0, 0, 0.25)', boxShadow: 'unset' }}
                    >
                        <InputBase
                            name="name"
                            sx={{ ml: 1, flex: 1 }}
                            placeholder={window.i18n("searchMycroft")}
                            inputProps={{ 'aria-label': 'search mycroft' }}
                        />
                        <IconButton type="submit" sx={{ p: '10px' }} aria-label="search">
                        <SearchIcon />
                        </IconButton>
                    </Paper>
                    <TableContainer component={Paper}>
                        <Table sx={{ minWidth: 650 }} aria-label="simple table">
                            <TableHead>
                                <TableRow>
                                    <TableCell>{window.i18n("param")}</TableCell>
                                    <TableCell>{window.i18n("details")}</TableCell>
                            </TableRow>
                            </TableHead>
                            <TableBody>
                            {rows.map((row) => (
                                <TableRow
                                  key={row.param}
                                  sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                                >
                                    <TableCell component="th" scope="row">
                                      {row.param}
                                    </TableCell>
                                    <TableCell>{row.info}</TableCell>
                                </TableRow>
                            ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}