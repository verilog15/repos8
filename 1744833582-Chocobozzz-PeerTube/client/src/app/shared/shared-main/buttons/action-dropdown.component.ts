import { ChangeDetectionStrategy, Component, input, output } from '@angular/core'
import { Params, RouterLink } from '@angular/router'
import { GlobalIconName } from '@app/shared/shared-icons/global-icon.component'
import { GlobalIconComponent } from '../../shared-icons/global-icon.component'
import { NgbDropdown, NgbDropdownToggle, NgbDropdownMenu, NgbTooltip } from '@ng-bootstrap/ng-bootstrap'
import { NgIf, NgClass, NgFor, NgTemplateOutlet } from '@angular/common'

export type DropdownAction<T> = {
  label?: string
  iconName?: GlobalIconName
  description?: string
  title?: string
  handler?: (a: T) => any

  linkBuilder?: (a: T) => (string | number)[]
  queryParamsBuilder?: (a: T) => Params

  isDisplayed?: (a: T) => boolean

  class?: string[]
  isHeader?: boolean

  ownerOrModeratorPrivilege?: () => string
}

export type DropdownButtonSize = 'normal' | 'small'
export type DropdownTheme = 'primary' | 'secondary'
export type DropdownDirection = 'horizontal' | 'vertical'

@Component({
  selector: 'my-action-dropdown',
  styleUrls: [ './action-dropdown.component.scss' ],
  templateUrl: './action-dropdown.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    NgIf,
    NgbTooltip,
    NgbDropdown,
    NgbDropdownToggle,
    NgClass,
    GlobalIconComponent,
    NgbDropdownMenu,
    NgFor,
    RouterLink,
    NgTemplateOutlet
  ]
})
export class ActionDropdownComponent<T> {
  readonly actions = input<DropdownAction<T>[] | DropdownAction<T>[][]>([])
  readonly entry = input<T>(undefined)

  readonly placement = input('bottom-left auto')
  readonly container = input<null | 'body'>(undefined)

  readonly buttonSize = input<DropdownButtonSize>('normal')
  readonly buttonDirection = input<DropdownDirection>('horizontal')
  readonly buttonStyled = input(true)

  readonly label = input<string>(undefined)
  readonly theme = input<DropdownTheme>('secondary')

  readonly openChange = output<boolean>()

  getActions (): DropdownAction<T>[][] {
    const actions = this.actions()
    if (actions.length !== 0 && Array.isArray(actions[0])) return actions as DropdownAction<T>[][]

    return [ actions as DropdownAction<T>[] ]
  }

  getQueryParams (action: DropdownAction<T>, entry: T) {
    if (action.queryParamsBuilder) return action.queryParamsBuilder(entry)

    return {}
  }

  areActionsDisplayed (actions: (DropdownAction<T> | DropdownAction<T>[])[], entry: T): boolean {
    return actions.some(a => {
      if (Array.isArray(a)) return this.areActionsDisplayed(a, entry)

      return a.isHeader !== true && (a.isDisplayed === undefined || a.isDisplayed(entry))
    })
  }

  isBlockDisplayed (allActions: (DropdownAction<T> | DropdownAction<T>[])[], action: DropdownAction<T>, entry: T) {
    // Do not display only the header
    if (action.isHeader && !this.areActionsDisplayed(allActions, entry)) return false

    return action.isDisplayed === undefined || action.isDisplayed(entry) === true
  }
}
